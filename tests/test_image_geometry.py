import pytest

from vdgr_nnunet.image_geometry import ImageGeometry, require_matching_geometry


REFERENCE = ImageGeometry(
    size_xyz=(32, 24, 16),
    spacing_xyz=(0.7, 0.8, 1.2),
    origin_xyz=(-120.0, -100.0, -80.0),
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)


def test_matching_geometry_is_accepted() -> None:
    require_matching_geometry(REFERENCE, REFERENCE, context="test image")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("size_xyz", (31, 24, 16)),
        ("spacing_xyz", (0.7, 0.8, 1.3)),
        ("origin_xyz", (-119.0, -100.0, -80.0)),
        ("direction", (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    ],
)
def test_geometry_mismatch_is_rejected(field: str, replacement: tuple) -> None:
    values = {
        "size_xyz": REFERENCE.size_xyz,
        "spacing_xyz": REFERENCE.spacing_xyz,
        "origin_xyz": REFERENCE.origin_xyz,
        "direction": REFERENCE.direction,
    }
    values[field] = replacement
    candidate = ImageGeometry(**values)

    with pytest.raises(ValueError, match=field.removesuffix("_xyz")):
        require_matching_geometry(REFERENCE, candidate, context="test image")
