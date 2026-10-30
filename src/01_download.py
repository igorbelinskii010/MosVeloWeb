"""Шаг 1. Скачиваем данные: граница ЦАО, граф улиц и точки притяжения.

Что берёт: настройки из config/download.yaml и данные OpenStreetMap (через OSMnx).
Что сохраняет в data/processed/:
  cao_boundary.gpkg  - граница ЦАО (слои "cao" и "cao_buffer")
  graph_raw.graphml  - граф улиц (ЦАО + буфер) как скачан
  poi.gpkg           - точки притяжения с колонкой category
Скачанное кэшируется в data/raw/, поэтому повторный запуск не качает всё заново.
"""
from pathlib import Path

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
for folder in (RAW, PROCESSED, MAPS):
    folder.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "download.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Кэш запросов к Overpass лежит в data/raw/, чтобы не качать повторно
ox.settings.cache_folder = str(RAW / "osmnx_cache")
ox.settings.use_cache = True
ox.settings.useful_tags_way = cfg["useful_tags"]
# Какой сервер Overpass использовать и сколько ждать ответа (меняется в конфиге)
ox.settings.overpass_url = cfg["overpass_url"]
ox.settings.requests_timeout = cfg["overpass_timeout_s"]
ox.settings.overpass_rate_limit = cfg["overpass_rate_limit"]
# Печатать в консоль, что делает OSMnx (видно, если сервер тормозит или просит подождать)
ox.settings.log_console = True
print(f"Сервер Overpass: {cfg['overpass_url']}")


def get_boundary():
    """Граница ЦАО и такой же полигон с буфером. Проверяем площадь."""
    gdf = ox.geocode_to_gdf(cfg["place_query"])
    metric = gdf.to_crs(cfg["metric_crs"])
    area_km2 = metric.geometry.area.iloc[0] / 1e6
    low, high = cfg["expected_area_km2"]
    print(f"Площадь найденной территории: {area_km2:.1f} км2")
    if not low <= area_km2 <= high:
        raise SystemExit(f"Площадь вне диапазона {low}-{high} км2: нашлось не то. Проверь place_query.")
    buffered = metric.copy()
    # Буфер + упрощение контура: у границы ЦАО тысячи точек, а длинный контур
    # делает запрос к Overpass огромным и сервер не успевает его обработать
    buffered["geometry"] = metric.geometry.buffer(cfg["buffer_m"]).simplify(cfg["simplify_m"])
    return gdf[["geometry"]].reset_index(drop=True), buffered[["geometry"]].to_crs(4326).reset_index(drop=True)


def download_graph(polygon):
    """Граф улиц: берём все highway, кроме исключённых типов."""
    exclude = "|".join(cfg["exclude_highway"])
    custom_filter = f'["highway"]["highway"!~"{exclude}"]'
    raw_file = RAW / "graph_raw.graphml"
    if raw_file.exists():
        print("Граф уже скачан, читаю из data/raw/")
        return ox.load_graphml(raw_file)
    print("Качаю граф улиц (может занять несколько минут)...")
    graph = ox.graph_from_polygon(polygon, custom_filter=custom_filter, simplify=True, retain_all=True)
    ox.save_graphml(graph, raw_file)
    return graph


def download_poi(polygon):
    """Точки притяжения по категориям из конфига. Полигоны превращаем в центроиды."""
    raw_file = RAW / "poi_raw.gpkg"
    if raw_file.exists():
        print("Точки притяжения уже скачаны, читаю из data/raw/")
        return gpd.read_file(raw_file)
    parts = []
    for category, tags in cfg["poi"].items():
        print(f"Качаю точки: {category}")
        found = ox.features_from_polygon(polygon, tags=tags).reset_index()
        # Станции с train=yes считаем вокзалами, а не метро
        if category == "metro_mcd" and "train" in found.columns:
            found = found[found["train"] != "yes"]
        if "name" not in found.columns:
            found["name"] = None
        found = found[["element", "id", "name", "geometry"]].rename(columns={"id": "osm_id"})
        found["category"] = category
        parts.append(found)
    poi = pd.concat(parts, ignore_index=True)
    poi = gpd.GeoDataFrame(poi, geometry="geometry", crs=4326)
    # Центроид считаем в метрической системе, потом возвращаем в широту/долготу
    poi["geometry"] = poi.to_crs(cfg["metric_crs"]).centroid.to_crs(4326)
    poi.to_file(raw_file, driver="GPKG")
    return poi


def main():
    cao, cao_buffer = get_boundary()
    cao.to_file(PROCESSED / "cao_boundary.gpkg", layer="cao", driver="GPKG")
    cao_buffer.to_file(PROCESSED / "cao_boundary.gpkg", layer="cao_buffer", driver="GPKG")
    polygon = cao_buffer.geometry.iloc[0]

    graph = download_graph(polygon)
    ox.save_graphml(graph, PROCESSED / "graph_raw.graphml")
    poi = download_poi(polygon)
    poi.to_file(PROCESSED / "poi.gpkg", driver="GPKG")

    # Сводка для проверки глазами
    nodes, edges = ox.graph_to_gdfs(graph)
    total_km = edges.to_crs(cfg["metric_crs"]).geometry.length.sum() / 1000
    print(f"\nУзлов: {len(nodes)}, рёбер: {len(edges)}, длина сети: {total_km:.0f} км (вместе с буфером)")
    print("Точки притяжения по категориям:")
    print(poi["category"].value_counts().to_string())

    # Быстрая карта: граница, буфер, улицы, точки
    fmap = folium.Map(location=[55.755, 37.62], zoom_start=13, tiles="cartodbpositron")
    folium.GeoJson(cao_buffer, name="ЦАО + буфер", style_function=lambda x: {"color": "gray", "fill": False}).add_to(fmap)
    folium.GeoJson(cao, name="ЦАО", style_function=lambda x: {"color": "red", "fill": False}).add_to(fmap)
    folium.GeoJson(edges[["geometry"]], name="улицы", style_function=lambda x: {"color": "blue", "weight": 1}).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(MAPS / "download_check.html")
    print(f"\nКарта для проверки: {MAPS / 'download_check.html'}")


if __name__ == "__main__":
    main()
