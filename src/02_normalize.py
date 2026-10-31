"""Шаг 2. Приводим теги OpenStreetMap к единому виду.

Что берёт: data/processed/graph_raw.graphml и cao_boundary.gpkg (шаг 1),
           правила из config/lts_rules.yaml.
Что сохраняет в data/processed/:
  edges_norm.gpkg      - участки улиц с понятными колонками (скорость, полосы, тип велоинфраструктуры)
  edges_excluded.gpkg  - то, что мы убрали, с причиной (чтобы посмотреть глазами на карте)
Правила - метод v0.2 (config/lts_rules.yaml).
Сводку с цифрами печатает в консоль и пишет в outputs/normalize_summary.txt.

Главные идеи:
  1. В графе каждая двусторонняя улица записана дважды (туда и обратно). Оставляем одну запись.
  2. После упрощения графа в тегах бывают списки (['40', '60']) - берём худший случай (максимум).
  3. Если тега нет, подставляем значение по умолчанию и пишем об этом в колонке *_source.
"""
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "download.yaml", encoding="utf-8") as f:
    DL = yaml.safe_load(f)
with open(ROOT / "config" / "lts_rules.yaml", encoding="utf-8") as f:
    RULES = yaml.safe_load(f)
METRIC = DL["metric_crs"]


# ---------- Мелкие помощники для тегов ----------

def as_list(value):
    """Любое значение тега -> список строк. Пустое значение -> пустой список."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def parse_speed(text):
    """Одно значение maxspeed -> км/ч. Если не поняли - None."""
    text = text.strip()
    if text in RULES["maxspeed_names"]:
        return RULES["maxspeed_names"][text]
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)?$", text)
    if match:
        speed = float(match.group(1))
        return round(speed * 1.609) if match.group(2) else round(speed)
    return None


def parse_int_max(values):
    """Список значений lanes -> максимум среди тех, что похожи на число."""
    numbers = []
    for text in values:
        for part in text.split(";"):
            if re.match(r"^\d+$", part.strip()):
                numbers.append(int(part))
    return max(numbers) if numbers else None


def parse_width(values):
    """Список значений width -> ширина в метрах. Берём минимум (худший случай). Нет числа -> NaN."""
    numbers = []
    for text in values:
        match = re.match(r"^(\d+(?:[.,]\d+)?)\s*(m|м)?$", text.strip())
        if match:
            numbers.append(float(match.group(1).replace(",", ".")))
    return min(numbers) if numbers else np.nan


def main_highway(values):
    """Если у участка несколько типов дороги, берём самый "автомобильный" по списку из конфига."""
    priority = RULES["highway_priority"]
    values = as_list(values)
    known = [v for v in values if v in priority]
    if known:
        return min(known, key=priority.index)
    return values[0] if values else "unknown"


def cycleway_values(row):
    """Все значения тегов cycleway, cycleway:left, cycleway:right, cycleway:both."""
    result = []
    for tag in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
        result += as_list(row.get(tag))
    return result


# ---------- Определение типа велоинфраструктуры (шаг А из block1.md) ----------

def get_bike_infra(row, highway):
    cycle = cycleway_values(row)
    bicycle = as_list(row.get("bicycle"))
    if highway == "cycleway" or "track" in cycle:
        return "separated"
    if highway in ("path", "footway") and "designated" in bicycle:
        return "separated"
    if "lane" in cycle:
        return "lane"
    if "share_busway" in cycle:
        return "bus_lane"
    if highway == "pedestrian" or highway in ("path", "footway"):
        # тротуары потом получат свой тип "sidewalk" (см. normalize)
        return "shared_path"
    return "none"


# ---------- Что исключаем ----------

def exclusion_reason(row, highway):
    """Причина, по которой участок не входит в сеть, или None, если он остаётся."""
    bicycle = as_list(row.get("bicycle"))
    bike_ok = any(v in RULES["allow_bicycle"] for v in bicycle)
    if highway in RULES["exclude_highway"]:
        return "тип дороги"
    if not bike_ok and any(v in RULES["exclude_bicycle"] for v in bicycle):
        return "bicycle=no"
    if not bike_ok and any(v in RULES["exclude_access"] for v in as_list(row.get("access"))):
        return "access=private/no"
    if any(v in RULES["exclude_service"] for v in as_list(row.get("service"))):
        return "проезд парковки (service=parking_aisle/drive-through)"
    # Тротуары НЕ исключаем, а помечаем (is_sidewalk): метод v0.2, вариант Т3 «две оценки»
    return None


def nearest_parallel_road(edges_m, targets):
    """Для каждой линии из targets ищем ближайшую проезжую часть, если она рядом и параллельна.

    Рядом = середина линии не дальше max_distance_m от дороги, параллельна = угол до max_angle_deg.
    Возвращает словарь {индекс линии: тип этой дороги (highway)}.
    """
    cfg = RULES["sidewalk_by_geometry"]
    roads = edges_m[edges_m["highway"].isin(cfg["road_types"])]
    road_geoms = roads.geometry.values
    road_types = roads["highway"].values
    tree = shapely.STRtree(road_geoms)
    found = {}
    for idx, geom in targets.geometry.items():
        mid = shapely.line_interpolate_point(geom, 0.5, normalized=True)
        near = tree.nearest(mid)
        road = road_geoms[near]
        if road.distance(mid) > cfg["max_distance_m"]:
            continue
        # направление дороги рядом с нашей точкой: смотрим на 5 м вперёд и назад
        pos = road.project(mid)
        p1 = road.interpolate(max(pos - 5, 0))
        p2 = road.interpolate(min(pos + 5, road.length))
        road_vec = np.array([p2.x - p1.x, p2.y - p1.y])
        # направление нашей линии: от начала к концу
        start, end = geom.coords[0], geom.coords[-1]
        foot_vec = np.array([end[0] - start[0], end[1] - start[1]])
        norm = np.linalg.norm(road_vec) * np.linalg.norm(foot_vec)
        if norm == 0:
            continue
        cos_angle = abs(np.dot(road_vec, foot_vec)) / norm
        if cos_angle >= np.cos(np.radians(cfg["max_angle_deg"])):
            found[idx] = road_types[near]
    return found


def find_geometric_sidewalks(edges_m):
    """Безымянные тротуары: highway=footway без тега footway, идут рядом и параллельно дороге.

    Возвращает набор индексов таких участков.
    """
    cfg = RULES["sidewalk_by_geometry"]
    if not cfg["enabled"]:
        return set()
    # кандидаты: просто линия-footway/path без подписи и без bicycle=yes/designated
    is_candidate = (
        edges_m["highway"].isin(["footway", "path"])
        & edges_m["footway_tag"].eq("")
        & ~edges_m["bicycle_ok"]
        & (edges_m.geometry.length >= cfg["min_length_m"])
    )
    return set(nearest_parallel_road(edges_m, edges_m[is_candidate]))


# ---------- Основная часть ----------

def load_edges():
    """Граф -> таблица рёбер в метрах, без дублей "туда и обратно"."""
    # Файл большой (около 110 МБ), чтение занимает 1-3 минуты. Это не зависание.
    print("Читаю граф из graph_raw.graphml (1-3 минуты, ничего не нажимай)...", flush=True)
    graph = ox.load_graphml(PROCESSED / "graph_raw.graphml")
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index().to_crs(METRIC)
    if "service" not in edges.columns:
        # тег service добавлен в download.yaml для метода v0.2: без него парковки не убрать
        raise SystemExit("В графе нет тега service. Удали data/raw/graph_raw.graphml и запусти 01_download.py заново.")
    n_directed = len(edges)
    km_directed = edges.geometry.length.sum() / 1000

    # Двустороннюю улицу OSMnx записывает двумя рёбрами (u->v и v->u). Оставляем одно.
    edges["a"] = edges[["u", "v"]].min(axis=1)
    edges["b"] = edges[["u", "v"]].max(axis=1)
    edges["osm_text"] = edges["osmid"].astype(str)
    edges["len_round"] = edges.geometry.length.round(1)
    edges["is_forward"] = edges["reversed"].astype(str).eq("False")
    edges = edges.sort_values("is_forward", ascending=False)  # сначала "прямые" копии
    edges = edges.drop_duplicates(["a", "b", "osm_text", "len_round"]).sort_index()
    print(f"Рёбер в графе (с направлениями): {n_directed}, длина {km_directed:.0f} км")
    print(f"После удаления копий 'туда и обратно': {len(edges)}, длина {edges.geometry.length.sum() / 1000:.0f} км")
    return edges.reset_index(drop=True)


def normalize(edges, cao):
    """Считаем все нормализованные колонки."""
    print("Разбираю теги каждого участка (около минуты)...", flush=True)
    rows = []
    for _, row in edges.iterrows():
        highway = main_highway(row["highway"])
        bicycle = as_list(row.get("bicycle"))
        oneway = any(str(v) == "True" for v in as_list(row.get("oneway")))
        is_car_road = highway not in RULES["non_car_highway"]

        # --- скорость ---
        speed_values = [s for s in (parse_speed(t) for t in as_list(row.get("maxspeed"))) if s is not None]
        if not is_car_road:
            speed, speed_src = np.nan, "n/a"          # для пешеходных и велодорожек скорость не нужна
        elif speed_values:
            speed, speed_src = max(speed_values), "tag"   # худший случай
        else:
            speed = RULES["maxspeed_default_living_street"] if highway == "living_street" else RULES["maxspeed_default"]
            speed_src = "default"

        # --- полосы ---
        lanes_tag = parse_int_max(as_list(row.get("lanes")))
        if not is_car_road:
            lanes_total, lanes_src, lanes_dir = np.nan, "n/a", np.nan
        else:
            if lanes_tag is not None:
                lanes_total, lanes_src = lanes_tag, "tag"
            else:
                lanes_total = RULES["lanes_default"].get(highway, RULES["lanes_default_other"])
                if oneway:
                    lanes_total = max(lanes_total // 2, 1)   # у односторонней дороги по умолчанию вдвое меньше
                lanes_src = "default"
            # lanes в OSM - это полосы в обе стороны. У односторонних - в одну.
            lanes_dir = lanes_total if oneway else lanes_total / 2

        infra = get_bike_infra(row, highway)
        bike_ok = any(v in RULES["allow_bicycle"] for v in bicycle)
        is_sidewalk = (not bike_ok
                       and any(v in RULES["sidewalk_footway_values"] for v in as_list(row.get("footway"))))
        # Тихая улица: тип из списка и не больше 1 полосы в сторону (метод v0.2)
        is_quiet = (highway in RULES["quiet_highway"]
                    and is_car_road and lanes_dir <= RULES["quiet_max_lanes_per_dir"])
        # Защищённая велополоса: у полосы есть тег cycleway:*:separation (столбики, бордюр...)
        separation = []
        for tag in RULES["lane_separation_tags"]:
            separation += as_list(row.get(tag))
        lane_protected = infra == "lane" and any(v != "no" for v in separation)
        oneway_bike_no = "no" in as_list(row.get("oneway:bicycle"))
        both_ways = (not oneway) or oneway_bike_no or infra == "separated"

        rows.append({
            "highway": highway,
            "maxspeed_kmh": speed, "maxspeed_source": speed_src,
            "lanes_total": lanes_total, "lanes_source": lanes_src, "lanes_per_dir": lanes_dir,
            "oneway": oneway, "bike_both_ways": both_ways, "bike_infra": infra,
            "is_quiet": is_quiet, "lane_protected": lane_protected,
            # Тротуар по тегу footway=sidewalk (если велосипедам не разрешено явно)
            "is_sidewalk": is_sidewalk, "sidewalk_source": "tag" if is_sidewalk else None,
            # ширина самой линии (у тротуара - ширина тротуара)
            "line_width_m": parse_width(as_list(row.get("width"))),
            # ширина тротуара, записанная на самой улице (sidewalk:*:width)
            "sidewalk_width_m": parse_width(sum((as_list(row.get(t)) for t in (
                "sidewalk:width", "sidewalk:left:width", "sidewalk:right:width", "sidewalk:both:width")), [])),
            "footway_tag": ";".join(as_list(row.get("footway"))),
            "bicycle_ok": any(v in RULES["allow_bicycle"] for v in bicycle),
            "excl_reason": exclusion_reason(row, highway),
        })
    new = pd.DataFrame(rows, index=edges.index)
    result = edges[["u", "v", "key", "osmid", "name", "geometry"]].copy()
    result["osmid"] = result["osmid"].astype(str)
    result["name"] = result["name"].map(lambda v: ";".join(as_list(v)) or None)
    result["length_m"] = result.geometry.length.round(1)
    result = pd.concat([result, new], axis=1)

    # Тротуары без подписи: ищем по геометрии
    print("Ищу тротуары без подписи по геометрии...", flush=True)
    todo = result[result["excl_reason"].isna() & ~result["is_sidewalk"]]
    for idx in find_geometric_sidewalks(todo):
        result.loc[idx, ["is_sidewalk", "sidewalk_source"]] = [True, "geometry"]

    # У тротуара свой тип велоинфраструктуры и ширина.
    # Ширину берём из тега width самой линии. Тег sidewalk:*:width стоит на улице, а не на
    # отдельной линии тротуара, поэтому его сохраняем у улицы в колонке sidewalk_width_m.
    walk = result["is_sidewalk"]
    result.loc[walk, "bike_infra"] = "sidewalk"
    result["width_m"] = np.where(walk, result["line_width_m"], np.nan)
    result["width_source"] = np.where(walk & result["line_width_m"].notna(), "width", None)
    result = result.drop(columns=["line_width_m"])

    # Вдоль какой улицы идёт тротуар (ближайшая параллельная проезжая часть, та же логика)
    print("Ищу, вдоль какой улицы идёт каждый тротуар...", flush=True)
    network = result[result["excl_reason"].isna()]
    road_of = nearest_parallel_road(network, result[walk])
    result["road_highway"] = pd.Series(road_of, dtype=object).reindex(result.index)

    # Если ширины нет: тротуар вдоль trunk/primary считаем широким (решение автора 31.10.2026)
    wide_roads = [hw for hw, yes in RULES["sidewalk_width_default"].items() if yes]
    fill = walk & result["width_m"].isna() & result["road_highway"].isin(wide_roads)
    result.loc[fill, "width_m"] = RULES["sidewalks"]["wide_sidewalk_min_width_m"]
    result.loc[fill, "width_source"] = "default_by_road"

    # Лежит ли середина участка внутри ЦАО (иначе это буфер 300 м)
    mid = result.geometry.interpolate(0.5, normalized=True)
    result["in_cao"] = mid.within(cao)
    return result


def share_text(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "-"


def make_summary(all_edges, kept):
    """Сводка по ЦАО (без буфера) в виде списка строк."""
    out = []
    add = out.append
    in_all = all_edges[all_edges["in_cao"]]
    in_kept = kept[kept["in_cao"]]
    km = lambda df: df["length_m"].sum() / 1000

    add("СВОДКА ШАГА 2 (только внутри границы ЦАО, без буфера 300 м)")
    no_walks = in_kept[~in_kept["is_sidewalk"]]
    add(f"Всего улиц и дорожек в OSM: {km(in_all):.0f} км, оставлено: {km(in_kept):.0f} км")
    add(f"  сеть «по ПДД» (без тротуаров): {km(no_walks):.0f} км, тротуары: {km(in_kept) - km(no_walks):.0f} км")
    add("\nЧто убрали (км):")
    removed = in_all[in_all["excl_reason"].notna()].groupby("excl_reason")["length_m"].sum() / 1000
    for reason, value in removed.sort_values(ascending=False).items():
        add(f"  {reason}: {value:.0f}")

    add("\nТип велоинфраструктуры в оставшейся сети (км):")
    infra = in_kept.groupby("bike_infra")["length_m"].sum() / 1000
    for name, value in infra.sort_values(ascending=False).items():
        add(f"  {name}: {value:.1f} ({share_text(value, km(in_kept))})")

    add("\nТипы дорог в оставшейся сети (км):")
    for name, value in (in_kept.groupby("highway")["length_m"].sum() / 1000).sort_values(ascending=False).head(8).items():
        add(f"  {name}: {value:.0f}")

    # Скорость и полосы важны там, где едем вместе с машинами: infra = none / lane
    streets = in_kept[in_kept["bike_infra"].isin(["none", "lane"])]
    add(f"\nУлицы со смешанным движением (none/lane): {km(streets):.0f} км")
    for col, label in (("maxspeed_source", "скорость"), ("lanes_source", "полосы")):
        by_src = streets.groupby(col)["length_m"].sum() / 1000
        tag, default = by_src.get("tag", 0), by_src.get("default", 0)
        add(f"  {label}: из тега {tag:.0f} км ({share_text(tag, km(streets))}), "
            f"по умолчанию {default:.0f} км ({share_text(default, km(streets))})")
    default_share = streets.loc[streets["maxspeed_source"] == "default", "length_m"].sum() / streets["length_m"].sum()
    if default_share > 0.5:
        add("  ВНИМАНИЕ: больше 50% скоростей по умолчанию - записать как ограничение, кандидат для ML в блоке 4")

    # Тихие улицы (is_quiet): сколько их и сколько из них без тега скорости
    quiet = streets[streets["is_quiet"]]
    add(f"\nТихие улицы (residential/service/living_street, до 1 полосы в сторону): {km(quiet):.0f} км "
        f"({share_text(km(quiet), km(streets))} улиц со смешанным движением)")
    for name, part in quiet.groupby("highway"):
        no_tag = part.loc[part["maxspeed_source"] == "default", "length_m"].sum() / 1000
        add(f"  {name}: {km(part):.0f} км, из них без тега скорости {no_tag:.0f} км")

    lanes = in_kept[in_kept["bike_infra"] == "lane"]
    protected = lanes[lanes["lane_protected"]]
    add(f"\nВелополосы: {km(lanes):.1f} км, из них с физической защитой (cycleway:*:separation): {km(protected):.1f} км")
    parking = in_all[in_all["excl_reason"].fillna("").str.startswith("проезд парковки")]
    add(f"Проезды парковок (исключены): {km(parking):.1f} км")

    # Тротуары и их ширина (нужно для оценки «с тротуарами»)
    wide = RULES["sidewalks"]["wide_sidewalk_min_width_m"]
    walks = in_kept[in_kept["is_sidewalk"]]
    tagged_walks = walks[walks["width_m"].notna()]
    wide_walks = walks[walks["width_m"] >= wide]
    add(f"\nТротуары: {km(walks):.0f} км, ширина указана у {km(tagged_walks):.0f} км "
        f"({share_text(km(tagged_walks), km(walks))}), из них от {wide} м: {km(wide_walks):.0f} км "
        f"({share_text(km(wide_walks), km(walks))} всех тротуаров)")
    for source, part in walks.groupby("sidewalk_source"):
        add(f"  найдено по {source}: {km(part):.0f} км, с шириной {km(part[part['width_m'].notna()]):.0f} км")
    by_src = walks.groupby(walks["width_source"].fillna("нет"))["length_m"].sum() / 1000
    add("  источник ширины: " + ", ".join(f"{name} {value:.0f} км" for name, value in by_src.items()))
    by_road = walks.groupby(walks["road_highway"].fillna("нет дороги рядом"))["length_m"].sum() / 1000
    add("  вдоль каких улиц: " + ", ".join(f"{name} {value:.0f}" for name, value in by_road.sort_values(ascending=False).items()))
    on_street = in_kept[in_kept["sidewalk_width_m"].notna()]
    add(f"Улицы с шириной тротуара в теге sidewalk:*:width: {km(on_street):.1f} км")

    add("Скорость из тега по типам дорог (км):")
    tagged = streets[streets["maxspeed_source"] == "tag"].groupby("maxspeed_kmh")["length_m"].sum() / 1000
    add("  " + ", ".join(f"{int(s)} км/ч: {v:.0f}" for s, v in tagged.items()))
    return out


def main():
    cao = gpd.read_file(PROCESSED / "cao_boundary.gpkg", layer="cao").to_crs(METRIC).geometry.iloc[0]
    edges = load_edges()
    result = normalize(edges, cao)

    # Сохраняем: оставленные и убранные отдельно
    helper = ["footway_tag", "bicycle_ok"]
    kept = result[result["excl_reason"].isna()].drop(columns=["excl_reason"] + helper)
    removed = result[result["excl_reason"].notna()].drop(columns=helper).rename(columns={"excl_reason": "excluded_reason"})
    print("Сохраняю результат...", flush=True)
    kept.to_file(PROCESSED / "edges_norm.gpkg", driver="GPKG")
    removed.to_file(PROCESSED / "edges_excluded.gpkg", driver="GPKG")

    summary = make_summary(result, result[result["excl_reason"].isna()])
    text = "\n".join(summary)
    print("\n" + text)
    (OUTPUTS / "normalize_summary.txt").write_text(text, encoding="utf-8")
    print(f"\nСохранено: {PROCESSED / 'edges_norm.gpkg'} ({len(kept)} участков), "
          f"edges_excluded.gpkg ({len(removed)} участков)")


if __name__ == "__main__":
    main()
