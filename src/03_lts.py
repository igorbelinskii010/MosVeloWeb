"""Шаг 3. Присваиваем каждому участку уровень стресса велосипедиста (LTS 1-4).

Что берёт: data/processed/edges_norm.gpkg (шаг 2) и правила из config/lts_rules.yaml
           (раздел «Уровни LTS», метод v0.2).
Что сохраняет в data/processed/:
  edges_lts.gpkg - те же участки + две оценки:
      lts, lts_reason                   - сеть «по ПДД» (у тротуаров пусто: по ПДД там ехать нельзя)
      lts_sidewalk, lts_sidewalk_reason - сеть «с тротуарами» (то же + тротуары)
  nodes.gpkg     - узлы сети (перекрёстки и концы участков) с числом примыкающих участков
Сводку «км по уровням LTS» печатает в консоль и пишет в outputs/lts_summary.txt.

Порядок правил (первое подходящее срабатывает):
  1. отдельная инфраструктура (велодорожка, пешеходная зона, автобусная полоса)
  2. велополоса на проезжей части
  3. тихая улица без велоинфраструктуры
  4. остальные улицы - таблица Mekuria (скорость x полосы)
  5. тротуар - только для оценки «с тротуарами»
"""
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "lts_rules.yaml", encoding="utf-8") as f:
    RULES = yaml.safe_load(f)


# ---------- Помощники для текста причины ----------

def speed_text(row):
    """'60 км/ч' или '60 км/ч (по умолч.)', если скорости не было в OSM."""
    mark = " (по умолч.)" if row["maxspeed_source"] == "default" else ""
    return f"{row['maxspeed_kmh']:.0f} км/ч{mark}"


def lanes_text(row, lanes):
    mark = " (по умолч.)" if row["lanes_source"] == "default" else ""
    return f"{lanes} пол./сторону{mark}"


def lanes_one_way(row):
    """Полос в одну сторону целым числом.

    lanes_per_dir бывает дробным: 3 полосы на двустороннюю улицу = 1,5 в сторону.
    Округляем вверх (1,5 -> 2): средняя полоса бывает то в одну, то в другую сторону,
    поэтому берём худший случай, как и со скоростью на шаге 2.
    """
    if RULES["lanes_per_dir_round_up"]:
        return max(1, math.ceil(row["lanes_per_dir"]))
    return max(1, round(row["lanes_per_dir"]))


# ---------- Правила LTS (каждое возвращает пару: уровень и причина) ----------

def lts_infra(row):
    """1. Отдельная инфраструктура: велодорожка, пешеходная зона, автобусная полоса."""
    infra = row["bike_infra"]
    if infra in RULES["infra_lts"]:
        return RULES["infra_lts"][infra], f"{infra}"
    return None


def lts_bike_lane(row):
    """2. Велополоса на проезжей части (велоруководство 2022, табл. 10 и 13; Mekuria, табл. 3)."""
    cfg = RULES["bike_lane"]
    speed = row["maxspeed_kmh"]
    lanes = lanes_one_way(row)
    about = f"lane, {speed_text(row)}, {lanes_text(row, lanes)}"
    if row["lane_protected"] and speed <= cfg["lane_protected_max_speed"]:
        return cfg["lane_protected_lts"], f"{about}, с защитой"
    if speed >= cfg["lane_fast_min_speed"] or lanes >= cfg["lane_fast_min_lanes"]:
        return cfg["lane_fast_lts"], about
    if row["is_quiet"]:
        return cfg["lane_local_lts"], f"{about}, тихая улица"
    if speed <= cfg["lane_local_max_speed"] and lanes <= 1:
        return cfg["lane_local_lts"], about
    # 50-60 км/ч или 2 полосы - здесь стоит УСТУПКА (lane_arterial_lts)
    return cfg["lane_arterial_lts"], about


def lts_quiet(row):
    """3. Тихая улица без велоинфраструктуры (Mekuria, сноска a к табл. 4; альбом ДТ, разд. 5.5.1)."""
    cfg = RULES["quiet_lts"]
    speed = row["maxspeed_kmh"]
    about = f"тихая {row['highway']}, {speed_text(row)}"
    if row["highway"] == "living_street":
        return cfg["slow_lts"], f"тихая living_street (жилая зона)"
    if row["maxspeed_source"] == "default":
        return cfg["default_lts"], about
    if speed <= cfg["slow_max_speed"]:
        return cfg["slow_lts"], about
    if speed <= cfg["medium_max_speed"]:
        return cfg["medium_lts"], about
    return cfg["fast_lts"], about


def lts_mixed(row):
    """4. Остальные улицы без велоинфраструктуры: таблица Mekuria 2012, табл. 4."""
    speed = row["maxspeed_kmh"]
    lanes = lanes_one_way(row)
    column = min(lanes, 3) - 1          # 1 полоса -> столбец 0, 2 -> 1, 3 и больше -> 2
    for line in RULES["mixed_lts"]:     # строки идут по возрастанию скорости
        if speed <= line["max_speed"]:
            level = line["lts"][column]
            return level, f"{row['highway']}, {speed_text(row)}, {lanes_text(row, lanes)}"
    raise ValueError(f"Скорость {speed} не попала ни в одну строку mixed_lts")


def lts_sidewalk(row):
    """5. Тротуар (только оценка «с тротуарами»): широкий -> 2, узкий или ширина неизвестна -> 3."""
    cfg = RULES["sidewalks"]
    width = row["width_m"]
    if pd.notna(width) and width >= cfg["wide_sidewalk_min_width_m"]:
        mark = " (по типу улицы)" if row["width_source"] == "default_by_road" else ""
        return cfg["wide_sidewalk_lts"], f"тротуар {width:.1f} м{mark}"
    if pd.notna(width):
        return cfg["sidewalk_lts"], f"тротуар {width:.1f} м (узкий)"
    return cfg["sidewalk_lts"], "тротуар, ширина неизвестна"


def street_lts(row):
    """Уровень для всего, кроме тротуаров (одинаковый в обеих оценках)."""
    if row["bike_infra"] == "lane":
        return lts_bike_lane(row)
    result = lts_infra(row)
    if result:
        return result
    if row["is_quiet"]:
        return lts_quiet(row)
    return lts_mixed(row)


# ---------- Основная часть ----------

def add_lts(edges):
    """Добавляем колонки двух оценок."""
    road, road_reason, walk, walk_reason = [], [], [], []
    for _, row in edges.iterrows():
        if row["is_sidewalk"]:
            # по ПДД на тротуаре ехать нельзя - в сеть «по ПДД» не входит
            road.append(None)
            road_reason.append("тротуар, по ПДД нельзя")
            level, reason = lts_sidewalk(row)
        else:
            level, reason = street_lts(row)
            road.append(level)
            road_reason.append(f"{reason} -> {level}")
        walk.append(level)
        walk_reason.append(f"{reason} -> {level}")
    edges["lts"] = pd.array(road, dtype="Int64")
    edges["lts_reason"] = road_reason
    edges["lts_sidewalk"] = pd.array(walk, dtype="Int64")
    edges["lts_sidewalk_reason"] = walk_reason
    return edges


def make_nodes(edges):
    """Узлы сети из концов участков: id узла OSM, точка, сколько участков сходится.

    degree_road считаем по сети «по ПДД» (без тротуаров), degree_all - по сети «с тротуарами».
    """
    points = {}
    for u, v, geom in zip(edges["u"], edges["v"], edges.geometry):
        # Линия участка идёт от узла u к узлу v
        points.setdefault(u, Point(geom.coords[0]))
        points.setdefault(v, Point(geom.coords[-1]))
    road = edges[~edges["is_sidewalk"]]
    degree_road = pd.concat([road["u"], road["v"]]).value_counts()
    degree_all = pd.concat([edges["u"], edges["v"]]).value_counts()
    nodes = gpd.GeoDataFrame({"osmid": list(points)}, geometry=list(points.values()), crs=edges.crs)
    nodes["degree_road"] = nodes["osmid"].map(degree_road).fillna(0).astype(int)
    nodes["degree_all"] = nodes["osmid"].map(degree_all).fillna(0).astype(int)
    return nodes


def km_by_level(part, column):
    """Таблица: уровень -> км и доля."""
    km = part.groupby(column)["length_m"].sum() / 1000
    total = km.sum()
    return [(int(level), value, 100 * value / total) for level, value in km.items()]


def make_summary(edges):
    in_cao = edges[edges["in_cao"]]
    out = []
    add = out.append
    add(f"СВОДКА ШАГА 3 (LTS v{RULES['version']}, только внутри границы ЦАО, без буфера 300 м)")

    for column, title in (("lts", "Сеть «по ПДД» (без тротуаров)"),
                          ("lts_sidewalk", "Сеть «с тротуарами»")):
        part = in_cao[in_cao[column].notna()]
        add(f"\n{title}: {part['length_m'].sum() / 1000:.0f} км")
        for level, value, share in km_by_level(part, column):
            add(f"  LTS {level}: {value:7.1f} км  ({share:4.1f}%)")
        calm = part[part[column] <= 2]["length_m"].sum() / 1000
        add(f"  спокойная сеть (LTS 1-2): {calm:.0f} км")

    # Откуда берётся каждый уровень: тип инфраструктуры x уровень (сеть «по ПДД»)
    road = in_cao[in_cao["lts"].notna()].copy()
    road["group"] = road["bike_infra"]
    road.loc[road["is_quiet"] & road["bike_infra"].eq("none"), "group"] = "none (тихая)"
    table = road.pivot_table(index="group", columns="lts", values="length_m", aggfunc="sum", fill_value=0) / 1000
    add("\nОткуда уровни, сеть «по ПДД» (км; строка - тип, столбец - LTS):")
    add("  " + "тип".ljust(14) + "".join(f"LTS {int(c)}".rjust(9) for c in table.columns))
    for name, line in table.iterrows():
        add("  " + str(name).ljust(14) + "".join(f"{v:9.1f}" for v in line))

    # Сколько километров каждого уровня держится на скорости «по умолчанию»
    streets = road[road["bike_infra"].isin(["none", "lane"])]
    add("\nУлицы с машинами: доля длины со скоростью по умолчанию (без тега maxspeed):")
    for level, part in streets.groupby("lts"):
        share = 100 * part.loc[part["maxspeed_source"] == "default", "length_m"].sum() / part["length_m"].sum()
        add(f"  LTS {int(level)}: {part['length_m'].sum() / 1000:.0f} км, из них по умолчанию {share:.0f}%")

    walks = in_cao[in_cao["is_sidewalk"]]
    add("\nТротуары (только сеть «с тротуарами»):")
    for level, value, share in km_by_level(walks, "lts_sidewalk"):
        add(f"  LTS {level}: {value:.1f} км ({share:.1f}%)")

    add("\nСамые частые причины (сеть «с тротуарами», км):")
    reasons = in_cao.groupby("lts_sidewalk_reason")["length_m"].sum().sort_values(ascending=False) / 1000
    for reason, value in reasons.head(15).items():
        add(f"  {value:7.1f}  {reason}")
    return out


def main():
    print("Читаю edges_norm.gpkg...", flush=True)
    edges = gpd.read_file(PROCESSED / "edges_norm.gpkg")
    # В GeoPackage да/нет иногда читается как 0/1 - возвращаем тип «да/нет»
    for column in ("is_sidewalk", "is_quiet", "lane_protected", "in_cao"):
        edges[column] = edges[column].fillna(False).astype(bool)
    print(f"Участков: {len(edges)}. Считаю LTS (около минуты)...", flush=True)
    edges = add_lts(edges)
    nodes = make_nodes(edges)

    print("Сохраняю результат...", flush=True)
    edges.to_file(PROCESSED / "edges_lts.gpkg", driver="GPKG")
    nodes.to_file(PROCESSED / "nodes.gpkg", driver="GPKG")

    text = "\n".join(make_summary(edges))
    print("\n" + text)
    (OUTPUTS / "lts_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nСохранено: {PROCESSED / 'edges_lts.gpkg'} ({len(edges)} участков), "
          f"nodes.gpkg ({len(nodes)} узлов), outputs/lts_summary.txt")


if __name__ == "__main__":
    main()
