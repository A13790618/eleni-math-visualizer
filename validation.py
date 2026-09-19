"""Request validation for the public visualization API."""

from __future__ import annotations

import math


class ValidationError(ValueError):
    """Raised when a client request does not match the public API contract."""


GEOMETRY_SHAPES = {
    "triangle", "circle", "quadrilateral", "square", "rectangle",
    "parallelogram", "rhombus", "trapezoid", "angle", "parallel_lines",
    "parallel", "cylinder", "cone", "pyramid", "square_pyramid", "prism",
    "cube", "sphere", "3d_shape",
}

GEOMETRY_FEATURES = {
    "altitude_from_a", "altitude_from_b", "altitude_from_c", "all_altitudes",
    "median_from_a", "median_from_b", "median_from_c", "all_medians",
    "bisector_from_a", "bisector_from_b", "bisector_from_c", "all_bisectors",
    "angle_bisector_from_a", "angle_bisector_from_b", "angle_bisector_from_c",
    "circumscribed_circle", "inscribed_circle", "right_angle", "right_angles",
    "diagonals", "radius_line", "diameter", "tangent", "tangent_at_a",
    "chord", "chord_ab", "central_angle", "inscribed_angle", "sector",
}

SOLID_DIMENSIONS = {
    "cylinder": {"radius", "height"},
    "cone": {"radius", "height"},
    "pyramid": {"base_width", "base_depth", "height"},
    "square_pyramid": {"base_width", "base_depth", "height"},
    "prism": {"width", "depth", "height"},
    "cube": {"side"},
    "sphere": {"radius"},
}


def _normalise_feature(value):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Every feature must be a non-empty string")
    return value.strip().lower().replace(" ", "_")


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def _point(value, field):
    # Accept both [x, y] arrays and {"x": .., "y": ..} objects. The object
    # form exists because some OpenAPI-import UIs (Coze's Test Run panel,
    # notably) cannot reliably build nested array-of-array controls and
    # silently submit a null placeholder instead of the typed values.
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ValidationError(f"{field} must contain x and y coordinates")
        return [_finite_number(value["x"], f"{field}.x"),
                _finite_number(value["y"], f"{field}.y")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValidationError(f"{field} must contain exactly two coordinates")
    return [_finite_number(value[0], f"{field}[0]"),
            _finite_number(value[1], f"{field}[1]")]


def _points(value, expected, field="vertices"):
    if not isinstance(value, list) or len(value) != expected:
        raise ValidationError(f"{field} must contain exactly {expected} points")
    return [_point(point, f"{field}[{index}]") for index, point in enumerate(value)]


def _labels(value, expected=None):
    if value is None:
        return None
    if isinstance(value, dict):
        cleaned = {}
        for key, label in value.items():
            if not isinstance(key, str) or not isinstance(label, str) or not label.strip():
                raise ValidationError("Label keys and values must be non-empty strings")
            cleaned[key] = label.strip()[:20]
        return cleaned
    if not isinstance(value, list) or any(not isinstance(label, str) or not label.strip()
                                          for label in value):
        raise ValidationError("labels must be a list of non-empty strings")
    if expected is not None and len(value) != expected:
        raise ValidationError(f"labels must contain exactly {expected} items")
    return [label.strip()[:20] for label in value]


def validate_geometry_request(data):
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")

    shape = data.get("shape")
    if not isinstance(shape, str) or not shape.strip():
        raise ValidationError("shape is required")
    shape = shape.strip().lower()
    if shape not in GEOMETRY_SHAPES:
        raise ValidationError(f"Unsupported shape: {shape}")

    result = dict(data)
    result["shape"] = shape

    raw_features = data.get("features", [])
    if not isinstance(raw_features, list) or len(raw_features) > 20:
        raise ValidationError("features must be a list with at most 20 items")
    features = [_normalise_feature(feature) for feature in raw_features]
    unknown = sorted(set(features) - GEOMETRY_FEATURES)
    if unknown:
        raise ValidationError(f"Unsupported feature(s): {', '.join(unknown)}")
    result["features"] = features

    title = data.get("title")
    if title is not None and (not isinstance(title, str) or len(title) > 160):
        raise ValidationError("title must be a string no longer than 160 characters")

    for field in ("show_grid", "show_axis", "use_greek"):
        if field in data and not isinstance(data[field], bool):
            raise ValidationError(f"{field} must be a boolean")

    if shape == "triangle":
        result["vertices"] = _points(data.get("vertices"), 3)
        a, b, c = result["vertices"]
        twice_area = abs((b[0] - a[0]) * (c[1] - a[1])
                         - (b[1] - a[1]) * (c[0] - a[0]))
        if twice_area <= 1e-10:
            raise ValidationError("triangle vertices must not be collinear")
        result["labels"] = _labels(data.get("labels"), 3)
    elif shape in {"quadrilateral", "square", "rectangle", "parallelogram", "rhombus", "trapezoid"}:
        result["vertices"] = _points(data.get("vertices"), 4)
        result["labels"] = _labels(data.get("labels"), 4)
    elif shape == "circle":
        result["center"] = _point(data.get("center", [0, 0]), "center")
        result["radius"] = _finite_number(data.get("radius"), "radius")
        if result["radius"] <= 0:
            raise ValidationError("radius must be greater than zero")
        result["labels"] = _labels(data.get("labels"))
        angles = data.get("point_angles", [45, 145, 255])
        if not isinstance(angles, list) or not 1 <= len(angles) <= 8:
            raise ValidationError("point_angles must contain between 1 and 8 angles")
        result["point_angles"] = [_finite_number(angle, "point_angles") for angle in angles]
        required_points = 1
        if set(features) & {"chord", "chord_ab", "central_angle", "sector"}:
            required_points = 2
        if "inscribed_angle" in features:
            required_points = 3
        if len(result["point_angles"]) < required_points:
            raise ValidationError(
                f"Selected circle features require at least {required_points} point_angles"
            )
        point_labels = data.get("point_labels")
        if point_labels is not None:
            if (not isinstance(point_labels, list)
                    or len(point_labels) != len(result["point_angles"])
                    or any(not isinstance(label, str) or not label.strip()
                           for label in point_labels)):
                raise ValidationError("point_labels must match point_angles and contain non-empty strings")
            result["point_labels"] = [label.strip()[:20] for label in point_labels]
    elif shape == "angle":
        result["vertex"] = _point(data.get("vertex", [0, 0]), "vertex")
        result["ray1_end"] = _point(data.get("ray1_end", [3, 0]), "ray1_end")
        result["ray2_end"] = _point(data.get("ray2_end", [2, 2.5]), "ray2_end")
        if result["ray1_end"] == result["vertex"] or result["ray2_end"] == result["vertex"]:
            raise ValidationError("angle rays must end away from the vertex")
        if "show_bisector" in data and not isinstance(data["show_bisector"], bool):
            raise ValidationError("show_bisector must be a boolean")
        angle_label = data.get("angle_label")
        if angle_label is not None and (not isinstance(angle_label, str)
                                        or not angle_label.strip()
                                        or len(angle_label) > 20):
            raise ValidationError("angle_label must be a non-empty string up to 20 characters")
    elif shape in {"parallel_lines", "parallel"}:
        angle = _finite_number(data.get("transversal_angle", 60), "transversal_angle")
        if not 5 <= angle <= 175 or abs(angle - 90) < 0.001:
            raise ValidationError("transversal_angle must be between 5 and 175 degrees and not 90")
        result["transversal_angle"] = angle
    else:
        solid = data.get("shape_type", shape)
        if not isinstance(solid, str):
            raise ValidationError("shape_type must be a string")
        solid = solid.strip().lower()
        if solid == "3d_shape":
            raise ValidationError("shape_type is required when shape is 3d_shape")
        if solid not in SOLID_DIMENSIONS:
            raise ValidationError(f"Unsupported solid shape: {solid}")
        dimensions = data.get("dimensions", {})
        if not isinstance(dimensions, dict):
            raise ValidationError("dimensions must be an object")
        allowed = SOLID_DIMENSIONS[solid]
        unknown_dimensions = sorted(set(dimensions) - allowed)
        if unknown_dimensions:
            raise ValidationError(f"Unsupported dimension(s) for {solid}: {', '.join(unknown_dimensions)}")
        cleaned_dimensions = {}
        for key, value in dimensions.items():
            cleaned_dimensions[key] = _finite_number(value, f"dimensions.{key}")
            if cleaned_dimensions[key] <= 0 or cleaned_dimensions[key] > 10000:
                raise ValidationError(f"dimensions.{key} must be greater than zero and at most 10000")
        result["shape_type"] = solid
        result["dimensions"] = cleaned_dimensions

    return result


def validate_plot_request(data):
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    formulas = data.get("formulas")
    if isinstance(formulas, str):
        formulas = [formulas]
    if not isinstance(formulas, list) or not 1 <= len(formulas) <= 4:
        raise ValidationError("formulas must contain between 1 and 4 expressions")
    if any(not isinstance(formula, str) or not formula.strip() or len(formula) > 200
           for formula in formulas):
        raise ValidationError("Each formula must be a non-empty string no longer than 200 characters")

    result = dict(data)
    result["formulas"] = [formula.strip() for formula in formulas]
    x_range = data.get("x_range", [-10, 10])
    if not isinstance(x_range, (list, tuple)) or len(x_range) != 2:
        raise ValidationError("x_range must contain exactly two numbers")
    x_range = [_finite_number(x_range[0], "x_range[0]"),
               _finite_number(x_range[1], "x_range[1]")]
    if x_range[0] >= x_range[1] or x_range[1] - x_range[0] > 10000:
        raise ValidationError("x_range must be increasing and span at most 10000")
    result["x_range"] = x_range

    if data.get("y_range") is not None:
        y_range = data["y_range"]
        if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
            raise ValidationError("y_range must contain exactly two numbers")
        y_range = [_finite_number(y_range[0], "y_range[0]"),
                   _finite_number(y_range[1], "y_range[1]")]
        if y_range[0] >= y_range[1]:
            raise ValidationError("y_range must be increasing")
        result["y_range"] = y_range

    show_features = data.get("show_features", [])
    if not isinstance(show_features, list) or any(not isinstance(item, str)
                                                   for item in show_features):
        raise ValidationError("show_features must be a list of strings")
    show_features = [item.strip().lower() for item in show_features]
    allowed_features = {"roots", "extrema", "asymptotes", "intercepts", "all"}
    unknown_features = sorted(set(show_features) - allowed_features)
    if unknown_features:
        raise ValidationError(f"Unsupported plot feature(s): {', '.join(unknown_features)}")
    result["show_features"] = show_features

    labels = data.get("labels")
    if labels is not None:
        labels = _labels(labels, len(formulas))
        result["labels"] = labels

    title = data.get("title")
    if title is not None and (not isinstance(title, str) or len(title) > 160):
        raise ValidationError("title must be a string no longer than 160 characters")
    if "show_grid" in data and not isinstance(data["show_grid"], bool):
        raise ValidationError("show_grid must be a boolean")

    shade = data.get("shade")
    if isinstance(shade, dict) and not any(v is not None for v in shade.values()):
        # Coze submits every schema field; an empty/null-only shade means "no shading".
        shade = None
        result["shade"] = None
    if shade is not None:
        if not isinstance(shade, dict):
            raise ValidationError("shade must be an object")
        unknown_shade = sorted(set(shade) - {"x_start", "x_end", "formula_indices"})
        if unknown_shade:
            raise ValidationError(f"Unsupported shade field(s): {', '.join(unknown_shade)}")
        cleaned_shade = {}
        start = _finite_number(shade.get("x_start", x_range[0]), "shade.x_start")
        end = _finite_number(shade.get("x_end", x_range[1]), "shade.x_end")
        if start >= end or start < x_range[0] or end > x_range[1]:
            raise ValidationError("shade range must be increasing and inside x_range")
        cleaned_shade["x_start"] = start
        cleaned_shade["x_end"] = end
        if "formula_indices" in shade:
            indices = shade["formula_indices"]
            if (not isinstance(indices, list) or len(indices) != 2
                    or any(isinstance(index, bool) or not isinstance(index, int)
                           for index in indices)
                    or len(set(indices)) != 2
                    or any(index < 0 or index >= len(formulas) for index in indices)):
                raise ValidationError("formula_indices must contain two distinct valid indices")
            cleaned_shade["formula_indices"] = indices
        result["shade"] = cleaned_shade

    return result
