"""Шаг 4. Карта для проверки оценок LTS глазами.

Что берёт (data/processed/):
  edges_lts.gpkg     - участки с двумя оценками LTS (шаг 3)
  poi.gpkg           - точки притяжения (шаг 1)
  cao_boundary.gpkg  - граница ЦАО (шаг 1)
Настройки карты (цвета, пунктир) - config/map.yaml.

Что сохраняет: outputs/maps/lts_check.html - карту, которая открывается в браузере.
На карте (слои включаются и выключаются справа вверху):
  - улицы сети «по ПДД», по слою на каждый уровень LTS (цвет = уровень);
  - тротуары (вместе с улицами это сеть «с тротуарами»), по умолчанию выключены;
  - пунктир = скорость или полосы взяты по умолчанию, оценка держится на допущении;
  - стыки спокойной сети с LTS 4 - для проверки правила барьера, по умолчанию выключены;
  - точки притяжения.
Клик по участку - окно с данными и причиной оценки, ссылка на линию в OSM.

Рисуем только участки внутри ЦАО (in_cao), как и цифры в работе: так карта легче.
"""
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

# Русские буквы в файле карты пишем как есть, а не кодами \uXXXX - иначе файл в 2-3 раза тяжелее
folium.GeoJson._template.environment.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": True}

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
MAPS.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "map.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# ---------- Подготовка данных ----------

def first_osm_id(osmid):
    """Первый id линии OSM: после упрощения графа бывает список '[123, 456]'."""
    return str(osmid).strip("[]").split(",")[0].strip()


def value_with_source(value, source, unit):
    """'60 км/ч' или '60 км/ч (по умолч.)'. Для дорожек значения нет - пишем прочерк."""
    if pd.isna(value):
        return "-"
    mark = " (по умолч.)" if source == "default" else ""
    return f"{value:g} {unit}{mark}"


def prepare_edges(edges):
    """Оставляем участки внутри ЦАО и готовим короткие колонки для всплывающих окон."""
    edges = edges[edges["in_cao"]].copy()

    # Пунктир: у улицы с машинами скорость или полосы взяты по умолчанию
    is_street = edges["bike_infra"].isin(CFG["dashed_for_infra"])
    has_default = (edges["maxspeed_source"] == "default") | (edges["lanes_source"] == "default")
    edges["dashed"] = is_street & has_default

    # Тексты для окна. Короткие латинские названия колонок и короткие значения - чтобы файл
    # карты был легче (участков 111 тысяч), русские подписи к ним - в ROAD_FIELDS ниже.
    edges["name_"] = edges["name"].fillna("")
    edges["speed"] = [value_with_source(v, s, "км/ч")
                      for v, s in zip(edges["maxspeed_kmh"], edges["maxspeed_source"])]
    edges["lanes"] = [value_with_source(v, s, "")
                      for v, s in zip(edges["lanes_total"], edges["lanes_source"])]
    edges["width"] = [
        "-" if pd.isna(w) else f"{w:g} м" + (" (по типу улицы)" if s == "default_by_road" else "")
        for w, s in zip(edges["width_m"], edges["width_source"])
    ]
    # Только номер линии: ссылка целиком у каждого участка добавила бы к файлу ~10 МБ
    edges["osm"] = edges["osmid"].map(first_osm_id)
    edges["lts"] = edges["lts"].astype("Int64")
    edges["lts_sidewalk"] = edges["lts_sidewalk"].astype(int)

    # Для карты нужны градусы (EPSG:4326). Округляем до 5 знаков (~1 м), чтобы файл был легче
    edges = edges.to_crs(4326)
    edges["geometry"] = shapely.transform(edges.geometry.values, lambda xy: np.round(xy, 5))
    return edges


OSM_ALIAS = "Линия OSM"

# Что показываем во всплывающем окне: колонка -> подпись (порядок как в задании шага 4)
ROAD_FIELDS = {
    "name_": "Название", "highway": "Тип (highway)", "speed": "Скорость",
    "lanes": "Полос всего", "bike_infra": "Велоинфраструктура",
    "lts": "LTS по ПДД", "lts_reason": "Причина", "osm": OSM_ALIAS,
}
SIDEWALK_FIELDS = {
    "name_": "Название", "bike_infra": "Велоинфраструктура", "width": "Ширина тротуара",
    "lts_sidewalk": "LTS с тротуарами", "lts_sidewalk_reason": "Причина", "osm": OSM_ALIAS,
}


def find_junctions(edges):
    """Стыки: узлы, где спокойный участок сети «по ПДД» (LTS 1-2) касается участка LTS 4.

    Это переходы (дорожка-переход кончается в узле на проезжей части) и перекрёстки
    спокойных улиц с большими. Светофоров в наших данных нет - их смотрим в OSM по ссылке.
    """
    road = edges[edges["lts"].notna()]
    calm = road[road["lts"] <= 2]
    busy = road[road["lts"] == 4]

    # Для каждого узла: какие участки LTS 4 в него входят (названия улиц)
    busy_names = {}
    for u, v, name in zip(busy["u"], busy["v"], busy["name_"]):
        for node in (u, v):
            busy_names.setdefault(node, set()).add(name or "без названия")

    # Для каждого такого узла: какие спокойные участки в него входят
    calm_kinds = {}
    node_point = {}
    for u, v, infra, geom in zip(calm["u"], calm["v"], calm["bike_infra"], calm.geometry):
        for node, xy in ((u, geom.coords[0]), (v, geom.coords[-1])):
            if node in busy_names:
                calm_kinds.setdefault(node, set()).add(infra)
                node_point[node] = xy

    rows = []
    for node, kinds in calm_kinds.items():
        # Если в узел входит дорожка (shared_path/separated) - скорее всего, это переход
        if kinds & {"shared_path", "separated"}:
            kind = "переход (дорожка через улицу)"
        else:
            kind = "перекрёсток со спокойной улицей"
        rows.append({
            "osmid": node,
            "Что": kind,
            "Улица LTS 4": ", ".join(sorted(busy_names[node])),
            "Спокойные участки": ", ".join(sorted(kinds)),
            "Узел OSM": f'<a href="https://www.openstreetmap.org/node/{node}" target="_blank">{node}</a>',
            "geometry": shapely.Point(node_point[node]),
        })
    return gpd.GeoDataFrame(rows, crs=4326)


# ---------- Слои карты ----------

def add_lines(fmap, part, name, lts_column, fields, show):
    """Слой линий: цвет по уровню LTS, пунктир для оценок по умолчанию, окно по клику."""
    if part.empty:
        return
    colors = CFG["lts_colors"]

    def style(feature):
        props = feature["properties"]
        return {
            "color": colors[int(props[lts_column])],
            "weight": CFG["line_weight"],
            "opacity": 0.9,
            "dashArray": CFG["dash_pattern"] if props["dashed"] else None,
        }

    columns = list(dict.fromkeys(list(fields) + [lts_column, "dashed", "geometry"]))
    folium.GeoJson(
        part[columns],
        name=name,
        show=show,
        style_function=style,
        highlight_function=lambda feature: {"weight": CFG["line_weight"] + 4},
        tooltip=folium.GeoJsonTooltip(fields=["name_", lts_column], aliases=["", "LTS"]),
        popup=folium.GeoJsonPopup(fields=list(fields), aliases=list(fields.values()),
                                  style="min-width:320px"),
    ).add_to(fmap)


def add_junctions(fmap, junctions):
    group = folium.FeatureGroup(name=f"Стыки спокойной сети с LTS 4 ({len(junctions)})", show=False)
    folium.GeoJson(
        junctions,
        marker=folium.CircleMarker(radius=CFG["junction_radius"], color=CFG["junction_color"],
                                   fill=True, fill_opacity=0.8),
        popup=folium.GeoJsonPopup(fields=["Что", "Улица LTS 4", "Спокойные участки", "Узел OSM"],
                                  style="min-width:320px"),
    ).add_to(group)
    group.add_to(fmap)


def add_poi(fmap, poi):
    group = folium.FeatureGroup(name="Точки притяжения", show=True)
    for category, part in poi.groupby("category"):
        color = CFG["poi_colors"].get(category, "black")
        label = CFG["poi_names"].get(category, category)
        for name, point in zip(part["name"], part.geometry):
            folium.CircleMarker(
                location=[point.y, point.x], radius=5, color="white", weight=1,
                fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=f"{label}: {name if pd.notna(name) else 'без названия'}",
            ).add_to(group)
    group.add_to(fmap)


def add_legend(fmap):
    """Легенда в левом нижнем углу (простой HTML-блок)."""
    lines = "".join(
        f'<div><span style="display:inline-block;width:24px;height:4px;background:{color};'
        f'vertical-align:middle"></span> LTS {level}</div>'
        for level, color in CFG["lts_colors"].items()
    )
    html = f"""
    <div style="position:fixed;bottom:20px;left:10px;z-index:9999;background:white;
                padding:8px 10px;border:1px solid #999;border-radius:4px;font:13px sans-serif">
      <b>Уровень стресса</b>{lines}
      <div><span style="display:inline-block;width:24px;border-top:3px dashed #555;
           vertical-align:middle"></span> скорость/полосы по умолч.</div>
      <div style="margin-top:4px;color:#555">© участники OpenStreetMap</div>
    </div>"""
    fmap.get_root().html.add_child(folium.Element(html))


def main():
    print("Читаю edges_lts.gpkg...", flush=True)
    edges = gpd.read_file(PROCESSED / "edges_lts.gpkg")
    for column in ("is_sidewalk", "in_cao"):
        edges[column] = edges[column].fillna(False).astype(bool)
    edges = prepare_edges(edges)
    poi = gpd.read_file(PROCESSED / "poi.gpkg").to_crs(4326)
    cao = gpd.read_file(PROCESSED / "cao_boundary.gpkg", layer="cao").to_crs(4326)

    center = cao.geometry.iloc[0].centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=CFG["zoom"], tiles=None,
                      prefer_canvas=True)
    # Подложка OSM полупрозрачная, чтобы цветные линии было лучше видно
    folium.TileLayer("OpenStreetMap", name="Подложка OpenStreetMap",
                     opacity=CFG["tiles_opacity"]).add_to(fmap)
    folium.GeoJson(cao, name="Граница ЦАО",
                   style_function=lambda x: {"color": "#444", "weight": 2, "fill": False}).add_to(fmap)

    print("Рисую слои...", flush=True)
    road = edges[~edges["is_sidewalk"]]
    for level in (1, 2, 3, 4):
        part = road[road["lts"] == level]
        km = part["length_m"].sum() / 1000
        add_lines(fmap, part, f"По ПДД: LTS {level} ({km:.0f} км)", "lts", ROAD_FIELDS, show=True)
    walks = edges[edges["is_sidewalk"]]
    for level in sorted(walks["lts_sidewalk"].unique()):
        part = walks[walks["lts_sidewalk"] == level]
        km = part["length_m"].sum() / 1000
        add_lines(fmap, part, f"Тротуары: LTS {level} ({km:.0f} км)", "lts_sidewalk",
                  SIDEWALK_FIELDS, show=False)

    junctions = find_junctions(edges)
    add_junctions(fmap, junctions)
    add_poi(fmap, poi)
    add_legend(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = MAPS / "lts_check.html"
    fmap.save(out)

    # Короткая сводка в консоль
    dashed_km = edges.loc[edges["dashed"], "length_m"].sum() / 1000
    streets_km = edges.loc[edges["bike_infra"].isin(CFG["dashed_for_infra"]), "length_m"].sum() / 1000
    print(f"\nУчастков на карте (внутри ЦАО): {len(edges)}")
    print(f"Пунктир (скорость или полосы по умолч.): {dashed_km:.0f} км из {streets_km:.0f} км улиц с машинами")
    print(f"Стыков спокойной сети с LTS 4: {len(junctions)}")
    print(junctions["Что"].value_counts().to_string())
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nСохранено: {out} ({size_mb:.0f} МБ). Открой файл в браузере (лучше Chrome).")


if __name__ == "__main__":
    main()
