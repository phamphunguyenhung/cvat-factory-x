from shapely.geometry import Polygon, box

from cvat_tools.segmentation.stuff_things import ThingRegion, overlap_report, subtract_things

BUILDING = [0, 0, 100, 0, 100, 60, 0, 60]  # nền hình chữ nhật 100x60


def _area(flat):
    xs = flat[0::2]
    ys = flat[1::2]
    return Polygon(list(zip(xs, ys))).area


def test_truck_at_lower_boundary_cuts_notch():
    truck = ThingRegion("truck", box=(40, 40, 80, 80))  # đè mép dưới của building
    fix = subtract_things(BUILDING, [truck], box_inset=0.0)
    assert fix.changed and len(fix.polygons) == 1
    inter = box(40, 40, 80, 60).area
    assert abs(_area(fix.polygons[0]) - (6000 - inter)) < 2.0
    assert fix.overlaps[0]["label"] == "truck"


def test_thing_fully_inside_no_hole_stays_whole_region():
    truck = ThingRegion("truck", box=(30, 10, 60, 40))  # trọn trong building
    fix = subtract_things(BUILDING, [truck], box_inset=0.0)
    assert fix.changed
    assert len(fix.polygons) == 1
    assert abs(_area(fix.polygons[0]) - 6000) < 2.0  # nguyên vùng: không khoét lỗ
    p = Polygon(list(zip(fix.polygons[0][0::2], fix.polygons[0][1::2])))
    assert len(p.interiors) == 0


def test_small_overlap_below_threshold_is_ignored():
    truck = ThingRegion("truck", box=(0, 58, 100, 158))  # giao 200 / 10000 = 2% < 5%
    fix = subtract_things(BUILDING, [truck], box_inset=0.0)
    assert not fix.changed
    assert abs(_area(fix.polygons[0]) - 6000) < 1e-6


def test_non_thing_label_not_subtracted():
    sign = ThingRegion("traffic sign", box=(10, 10, 50, 50))
    assert overlap_report(BUILDING, [sign]) == []
    assert not subtract_things(BUILDING, [sign]).changed


def test_mask_polygon_preferred_over_box():
    poly = [40, 40, 80, 40, 80, 80, 40, 80]
    t = ThingRegion("car", box=(0, 0, 200, 200), polygon=poly, source="mask")
    fix = subtract_things(BUILDING, [t], box_inset=0.0)
    inter = box(40, 40, 80, 60).area
    assert abs(_area(fix.polygons[0]) - (6000 - inter)) < 2.0
    assert fix.overlaps[0]["source"] == "mask"
