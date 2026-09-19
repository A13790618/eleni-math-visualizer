"""
Geometry rendering module for Greek high school mathematics.
Supports: Triangle, Circle, Square, Rectangle, Parallelogram, Line, Segment, Angle.
Features: Altitudes, medians, bisectors, circumscribed/inscribed circles, labels in Greek.
"""

import io
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D

# Greek letter mapping for vertex labels
GREEK_LABELS = {
    'A': 'Α', 'B': 'Β', 'C': 'Γ', 'D': 'Δ', 'E': 'Ε',
    'F': 'Ζ', 'G': 'Η', 'H': 'Θ', 'I': 'Ι', 'K': 'Κ',
    'L': 'Λ', 'M': 'Μ', 'N': 'Ν', 'O': 'Ο', 'P': 'Π',
}

# Color palette
COLORS = {
    'shape_fill': '#E8F4FD',
    'shape_edge': '#2196F3',
    'auxiliary': '#FF5722',
    'auxiliary2': '#4CAF50',
    'auxiliary3': '#9C27B0',
    'point': '#1565C0',
    'label': '#0D47A1',
    'grid': '#E0E0E0',
    'axis': '#9E9E9E',
    'angle_fill': '#FFF3E0',
    'angle_edge': '#FF9800',
    'highlight': '#F44336',
}


def _setup_figure(title=None, show_grid=True, show_axis=True, figsize=(8, 8)):
    """Create a clean, professional matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    
    ax.set_aspect('equal', adjustable='box')
    
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    if show_axis:
        ax.axhline(y=0, color=COLORS['axis'], linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color=COLORS['axis'], linewidth=0.5, alpha=0.5)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15,
                     fontfamily='DejaVu Sans')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['axis'])
    ax.spines['left'].set_color(COLORS['axis'])
    
    return fig, ax


def _label_point(ax, point, label, offset=(0.15, 0.15), use_greek=True, fontsize=14):
    """Add a labeled point with optional Greek letter conversion."""
    display_label = GREEK_LABELS.get(label, label) if use_greek else label
    
    ax.plot(*point, 'o', color=COLORS['point'], markersize=6, zorder=5)
    ax.annotate(display_label, xy=point, xytext=offset, fontsize=fontsize,
                fontweight='bold', color=COLORS['label'],
                textcoords='offset points', zorder=6)


def _add_right_angle_mark(ax, vertex, p1, p2, size=0.3):
    """Draw a right angle mark at vertex between p1 and p2."""
    v = np.array(vertex)
    d1 = np.array(p1) - v
    d2 = np.array(p2) - v
    
    d1_norm = d1 / np.linalg.norm(d1) * size
    d2_norm = d2 / np.linalg.norm(d2) * size
    
    corner1 = v + d1_norm
    corner2 = v + d1_norm + d2_norm
    corner3 = v + d2_norm
    
    sq = plt.Polygon([corner1, corner2, corner3, v], fill=False,
                     edgecolor=COLORS['auxiliary'], linewidth=1.5, zorder=4)
    ax.add_patch(sq)


def _line_intersection(p1, p2, p3, p4):
    """Find intersection of line through p1-p2 and line through p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)


def _foot_of_perpendicular(point, line_p1, line_p2):
    """Find the foot of perpendicular from point to line defined by line_p1 and line_p2."""
    p = np.array(point)
    a = np.array(line_p1)
    b = np.array(line_p2)
    
    ab = b - a
    ap = p - a
    
    t = np.dot(ap, ab) / np.dot(ab, ab)
    foot = a + t * ab
    return tuple(foot)


def draw_triangle(vertices, labels=None, features=None, title=None, 
                  show_grid=True, show_axis=True, use_greek=True):
    """
    Draw a triangle with optional geometric features.
    
    Args:
        vertices: List of 3 (x, y) tuples
        labels: List of 3 label strings (default: ['A', 'B', 'C'])
        features: List of features to draw. Options:
            - 'altitude_from_A', 'altitude_from_B', 'altitude_from_C'
            - 'median_from_A', 'median_from_B', 'median_from_C'
            - 'bisector_from_A', 'bisector_from_B', 'bisector_from_C'
            - 'circumscribed_circle', 'inscribed_circle'
            - 'all_altitudes', 'all_medians', 'all_bisectors'
            - 'right_angle' (auto-detect or at specified vertex)
        title: Optional title
        show_grid, show_axis: Display options
        use_greek: Convert labels to Greek letters
    
    Returns:
        BytesIO object containing PNG image
    """
    if labels is None:
        labels = ['A', 'B', 'C']
    if features is None:
        features = []
    
    A, B, C = [np.array(v) for v in vertices]
    ab = B - A
    ac = C - A
    twice_area = abs(ab[0] * ac[1] - ab[1] * ac[0])
    if twice_area < 1e-9:
        raise ValueError('Triangle vertices must not be collinear')
    
    fig, ax = _setup_figure(title=title, show_grid=show_grid, show_axis=show_axis)
    
    # Draw triangle
    triangle = plt.Polygon([A, B, C], fill=True, facecolor=COLORS['shape_fill'],
                          edgecolor=COLORS['shape_edge'], linewidth=2.5, zorder=2)
    ax.add_patch(triangle)
    
    # Calculate label offsets based on centroid
    centroid = (A + B + C) / 3
    for pt, lbl in zip([A, B, C], labels):
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 0:
            offset = direction / norm * 18
        else:
            offset = np.array([15, 15])
        _label_point(ax, pt, lbl, offset=tuple(offset), use_greek=use_greek)
    
    vertex_map = {labels[0]: A, labels[1]: B, labels[2]: C}
    sides = [(labels[0], labels[1], labels[2]),  # A: opposite side BC
             (labels[1], labels[0], labels[2]),  # B: opposite side AC  
             (labels[2], labels[0], labels[1])]  # C: opposite side AB
    
    # Process features
    for feat in features:
        feat_lower = feat.lower().replace(' ', '_').replace('angle_bisector_', 'bisector_')
        
        # --- Altitudes ---
        if feat_lower in ['all_altitudes', 'altitude_from_a', 'altitude_from_b', 'altitude_from_c']:
            draw_indices = []
            if feat_lower == 'all_altitudes':
                draw_indices = [0, 1, 2]
            elif feat_lower == 'altitude_from_a':
                draw_indices = [0]
            elif feat_lower == 'altitude_from_b':
                draw_indices = [1]
            elif feat_lower == 'altitude_from_c':
                draw_indices = [2]
            
            alt_colors = [COLORS['auxiliary'], COLORS['auxiliary2'], COLORS['auxiliary3']]
            for i in draw_indices:
                vertex_label, opp1, opp2 = sides[i]
                vertex = vertex_map[vertex_label]
                foot = _foot_of_perpendicular(vertex, vertex_map[opp1], vertex_map[opp2])
                
                ax.plot([vertex[0], foot[0]], [vertex[1], foot[1]],
                       '--', color=alt_colors[i % 3], linewidth=1.5, zorder=3)
                _add_right_angle_mark(ax, foot, vertex, vertex_map[opp1])
                
                foot_label = f'H_{vertex_label}'
                _label_point(ax, foot, foot_label, offset=(8, -15), 
                           use_greek=False, fontsize=10)
        
        # --- Medians ---
        if feat_lower in ['all_medians', 'median_from_a', 'median_from_b', 'median_from_c']:
            draw_indices = []
            if feat_lower == 'all_medians':
                draw_indices = [0, 1, 2]
            elif feat_lower == 'median_from_a':
                draw_indices = [0]
            elif feat_lower == 'median_from_b':
                draw_indices = [1]
            elif feat_lower == 'median_from_c':
                draw_indices = [2]
            
            med_colors = [COLORS['auxiliary'], COLORS['auxiliary2'], COLORS['auxiliary3']]
            for i in draw_indices:
                vertex_label, opp1, opp2 = sides[i]
                vertex = vertex_map[vertex_label]
                midpoint = (vertex_map[opp1] + vertex_map[opp2]) / 2
                
                ax.plot([vertex[0], midpoint[0]], [vertex[1], midpoint[1]],
                       '-.', color=med_colors[i % 3], linewidth=1.5, zorder=3)
                
                mid_label = f'M_{vertex_label}'
                _label_point(ax, midpoint, mid_label, offset=(8, -15),
                           use_greek=False, fontsize=10)
            
            # Mark centroid if all medians
            if feat_lower == 'all_medians':
                _label_point(ax, centroid, 'G', offset=(10, 10),
                           use_greek=False, fontsize=12)
        
        # --- Angle bisectors ---
        if feat_lower in ['all_bisectors', 'bisector_from_a', 'bisector_from_b', 'bisector_from_c']:
            draw_indices = []
            if feat_lower == 'all_bisectors':
                draw_indices = [0, 1, 2]
            elif feat_lower == 'bisector_from_a':
                draw_indices = [0]
            elif feat_lower == 'bisector_from_b':
                draw_indices = [1]
            elif feat_lower == 'bisector_from_c':
                draw_indices = [2]
            
            bis_colors = [COLORS['auxiliary'], COLORS['auxiliary2'], COLORS['auxiliary3']]
            for i in draw_indices:
                vertex_label, opp1, opp2 = sides[i]
                vertex = vertex_map[vertex_label]
                
                d1 = vertex_map[opp1] - vertex
                d2 = vertex_map[opp2] - vertex
                d1_norm = d1 / np.linalg.norm(d1)
                d2_norm = d2 / np.linalg.norm(d2)
                bisector_dir = d1_norm + d2_norm
                
                # Find intersection with opposite side
                far_point = vertex + bisector_dir * 100
                foot = _line_intersection(vertex, far_point, vertex_map[opp1], vertex_map[opp2])
                
                if foot:
                    ax.plot([vertex[0], foot[0]], [vertex[1], foot[1]],
                           ':', color=bis_colors[i % 3], linewidth=1.5, zorder=3)
        
        # --- Circumscribed circle ---
        if feat_lower == 'circumscribed_circle':
            ax_val = A[0] - C[0]
            ay_val = A[1] - C[1]
            bx_val = B[0] - C[0]
            by_val = B[1] - C[1]
            
            D_val = 2 * (ax_val * by_val - ay_val * bx_val)
            if abs(D_val) > 1e-10:
                ux = (by_val * (ax_val**2 + ay_val**2) - ay_val * (bx_val**2 + by_val**2)) / D_val
                uy = (ax_val * (bx_val**2 + by_val**2) - bx_val * (ax_val**2 + ay_val**2)) / D_val
                center = np.array([ux + C[0], uy + C[1]])
                radius = np.linalg.norm(A - center)
                
                circle = plt.Circle(center, radius, fill=False,
                                   edgecolor=COLORS['auxiliary2'], linewidth=1.5,
                                   linestyle='--', zorder=3)
                ax.add_patch(circle)
                _label_point(ax, center, 'O', offset=(8, 8), use_greek=False, fontsize=11)
        
        # --- Inscribed circle ---
        if feat_lower == 'inscribed_circle':
            a_len = np.linalg.norm(B - C)  # side opposite A
            b_len = np.linalg.norm(A - C)  # side opposite B
            c_len = np.linalg.norm(A - B)  # side opposite C
            
            incenter = (a_len * A + b_len * B + c_len * C) / (a_len + b_len + c_len)
            s = (a_len + b_len + c_len) / 2
            inradius = math.sqrt(s * (s - a_len) * (s - b_len) * (s - c_len)) / s
            
            circle = plt.Circle(incenter, inradius, fill=True,
                               facecolor='#E8F5E9', edgecolor=COLORS['auxiliary2'],
                               linewidth=1.5, linestyle='--', zorder=3, alpha=0.5)
            ax.add_patch(circle)
            _label_point(ax, incenter, 'I', offset=(8, 8), use_greek=False, fontsize=11)

        if feat_lower in ['right_angle', 'right_angles']:
            triangle_points = [A, B, C]
            for index, vertex in enumerate(triangle_points):
                previous = triangle_points[(index - 1) % 3]
                following = triangle_points[(index + 1) % 3]
                d1 = previous - vertex
                d2 = following - vertex
                scale = np.linalg.norm(d1) * np.linalg.norm(d2)
                if scale and abs(np.dot(d1, d2)) <= 1e-6 * scale:
                    _add_right_angle_mark(ax, vertex, previous, following,
                                          size=max(np.linalg.norm(d1), np.linalg.norm(d2)) * 0.08)
    
    # Auto-adjust limits
    all_pts = np.array([A, B, C])
    margin = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.3
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def draw_circle(center, radius, labels=None, features=None, point_angles=None,
                point_labels=None, title=None, show_grid=True, show_axis=True,
                use_greek=True):
    """
    Draw a circle with optional features.
    
    Args:
        center: (x, y) tuple
        radius: float
        labels: dict with optional keys 'center', 'point_on_circle'
        features: List of features:
            - 'radius_line': draw a radius
            - 'diameter': draw a diameter
            - 'tangent_at_A': draw tangent at point A
            - 'chord_AB': draw chord from A to B
            - 'inscribed_angle': show inscribed angle
        title: Optional title
    
    Returns:
        BytesIO object containing PNG image
    """
    if labels is None:
        labels = {'center': 'O'}
    if features is None:
        features = []
    if point_angles is None:
        point_angles = [45, 145, 255]
    if point_labels is None:
        point_labels = ['A', 'B', 'C'][:len(point_angles)]
    
    O = np.array(center)
    r = radius
    circle_points = [
        O + np.array([r * math.cos(math.radians(angle)),
                      r * math.sin(math.radians(angle))])
        for angle in point_angles
    ]

    def circle_point(index, fallback_angle):
        if index < len(circle_points):
            return circle_points[index]
        return O + np.array([r * math.cos(math.radians(fallback_angle)),
                             r * math.sin(math.radians(fallback_angle))])

    def circle_label(index, fallback):
        return point_labels[index] if index < len(point_labels) else fallback
    
    fig, ax = _setup_figure(title=title, show_grid=show_grid, show_axis=show_axis)
    
    # Draw circle
    circle = plt.Circle(O, r, fill=True, facecolor=COLORS['shape_fill'],
                       edgecolor=COLORS['shape_edge'], linewidth=2.5, zorder=2)
    ax.add_patch(circle)
    
    # Label center
    _label_point(ax, O, labels.get('center', 'O'), offset=(8, 8), use_greek=use_greek)
    
    for feat in features:
        feat_lower = feat.lower().replace(' ', '_')
        
        if feat_lower == 'radius_line':
            end = circle_point(0, 45)
            ax.plot([O[0], end[0]], [O[1], end[1]], '-', color=COLORS['auxiliary'],
                   linewidth=2, zorder=3)
            mid = (O + end) / 2
            ax.annotate('r', xy=mid, fontsize=13, fontweight='bold',
                       color=COLORS['auxiliary'], ha='center')
            _label_point(ax, end, circle_label(0, 'A'), offset=(10, 10), use_greek=use_greek)
        
        if feat_lower == 'diameter':
            p1 = circle_point(0, 180)
            p2 = 2 * O - p1
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=COLORS['auxiliary'],
                   linewidth=2, zorder=3)
            _label_point(ax, p1, circle_label(0, 'A'), offset=(-15, -15), use_greek=use_greek)
            _label_point(ax, p2, circle_label(1, 'B'), offset=(10, -15), use_greek=use_greek)
        
        if 'tangent' in feat_lower:
            tangent_point = circle_point(0, 60)
            radius_direction = (tangent_point - O) / r
            # Tangent is perpendicular to radius
            tangent_dir = np.array([-radius_direction[1], radius_direction[0]])
            t1 = tangent_point - tangent_dir * r * 1.2
            t2 = tangent_point + tangent_dir * r * 1.2
            
            ax.plot([O[0], tangent_point[0]], [O[1], tangent_point[1]], '--', color=COLORS['auxiliary2'],
                   linewidth=1.5, zorder=3)
            ax.plot([t1[0], t2[0]], [t1[1], t2[1]], '-', color=COLORS['auxiliary'],
                   linewidth=2, zorder=3)
            _label_point(ax, tangent_point, circle_label(0, 'A'), offset=(10, 10), use_greek=use_greek)
            _add_right_angle_mark(ax, tangent_point, O, t2, size=r*0.12)

        if feat_lower in ['chord', 'chord_ab']:
            p1, p2 = circle_point(0, 25), circle_point(1, 155)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color=COLORS['auxiliary3'], linewidth=2, zorder=3)
            _label_point(ax, p1, circle_label(0, 'A'), offset=(10, 8), use_greek=use_greek)
            _label_point(ax, p2, circle_label(1, 'B'), offset=(-16, 8), use_greek=use_greek)

        if feat_lower in ['central_angle', 'sector']:
            p1, p2 = circle_point(0, 25), circle_point(1, 145)
            ax.plot([O[0], p1[0]], [O[1], p1[1]], '-', color=COLORS['auxiliary'], lw=1.8, zorder=3)
            ax.plot([O[0], p2[0]], [O[1], p2[1]], '-', color=COLORS['auxiliary'], lw=1.8, zorder=3)
            if feat_lower == 'sector':
                start, end_angle = point_angles[0] % 360, point_angles[1] % 360
                if end_angle < start:
                    end_angle += 360
                wedge = patches.Wedge(O, r, start, end_angle,
                                      facecolor=COLORS['angle_fill'],
                                      edgecolor=COLORS['angle_edge'], alpha=0.65,
                                      zorder=2.5)
                ax.add_patch(wedge)
            _label_point(ax, p1, circle_label(0, 'A'), offset=(10, 8), use_greek=use_greek)
            _label_point(ax, p2, circle_label(1, 'B'), offset=(-16, 8), use_greek=use_greek)

        if feat_lower == 'inscribed_angle':
            p1, p2 = circle_point(0, 20), circle_point(1, 150)
            vertex = circle_point(2, 270)
            ax.plot([vertex[0], p1[0]], [vertex[1], p1[1]], '-',
                    color=COLORS['auxiliary3'], linewidth=1.8, zorder=3)
            ax.plot([vertex[0], p2[0]], [vertex[1], p2[1]], '-',
                    color=COLORS['auxiliary3'], linewidth=1.8, zorder=3)
            _label_point(ax, p1, circle_label(0, 'A'), offset=(10, 8), use_greek=use_greek)
            _label_point(ax, p2, circle_label(1, 'B'), offset=(-16, 8), use_greek=use_greek)
            _label_point(ax, vertex, circle_label(2, 'C'), offset=(8, -18), use_greek=use_greek)
    
    # Auto-adjust limits
    margin = r * 0.6
    ax.set_xlim(O[0] - r - margin, O[0] + r + margin)
    ax.set_ylim(O[1] - r - margin, O[1] + r + margin)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def draw_quadrilateral(vertices, shape_type='quadrilateral', labels=None,
                       features=None, title=None, show_grid=True, show_axis=True,
                       use_greek=True):
    """
    Draw a quadrilateral (square, rectangle, parallelogram, rhombus, trapezoid).
    
    Args:
        vertices: List of 4 (x, y) tuples
        shape_type: 'square', 'rectangle', 'parallelogram', 'rhombus', 'trapezoid', 'quadrilateral'
        labels: List of 4 label strings
        features: List of features: 'diagonals', 'right_angles'
    """
    if labels is None:
        labels = ['A', 'B', 'C', 'D']
    if features is None:
        features = []
    
    pts = [np.array(v) for v in vertices]
    
    fig, ax = _setup_figure(title=title, show_grid=show_grid, show_axis=show_axis)
    
    # Draw quadrilateral
    quad = plt.Polygon(pts, fill=True, facecolor=COLORS['shape_fill'],
                      edgecolor=COLORS['shape_edge'], linewidth=2.5, zorder=2)
    ax.add_patch(quad)
    
    # Label vertices
    centroid = sum(pts) / 4
    for pt, lbl in zip(pts, labels):
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 0:
            offset = direction / norm * 18
        else:
            offset = np.array([15, 15])
        _label_point(ax, pt, lbl, offset=tuple(offset), use_greek=use_greek)
    
    for feat in features:
        if feat.lower() == 'diagonals':
            ax.plot([pts[0][0], pts[2][0]], [pts[0][1], pts[2][1]],
                   '--', color=COLORS['auxiliary'], linewidth=1.5, zorder=3)
            ax.plot([pts[1][0], pts[3][0]], [pts[1][1], pts[3][1]],
                   '--', color=COLORS['auxiliary2'], linewidth=1.5, zorder=3)
            
            inter = _line_intersection(pts[0], pts[2], pts[1], pts[3])
            if inter:
                _label_point(ax, inter, 'O', offset=(8, 8), use_greek=False, fontsize=11)
        
        if feat.lower() == 'right_angles':
            for i in range(4):
                _add_right_angle_mark(ax, pts[i], pts[(i-1) % 4], pts[(i+1) % 4])
    
    all_pts = np.array(pts)
    margin = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) * 0.3
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def draw_angle(vertex, ray1_end, ray2_end, label='α', show_arc=True,
               show_bisector=False, title=None, show_grid=True, show_axis=True):
    """Draw an angle with optional bisector."""
    V = np.array(vertex)
    P1 = np.array(ray1_end)
    P2 = np.array(ray2_end)
    
    fig, ax = _setup_figure(title=title, show_grid=show_grid, show_axis=show_axis)
    
    # Draw rays
    ray_length = max(np.linalg.norm(P1 - V), np.linalg.norm(P2 - V)) * 1.3
    d1 = (P1 - V) / np.linalg.norm(P1 - V)
    d2 = (P2 - V) / np.linalg.norm(P2 - V)
    
    end1 = V + d1 * ray_length
    end2 = V + d2 * ray_length
    
    ax.annotate('', xy=end1, xytext=V,
               arrowprops=dict(arrowstyle='->', color=COLORS['shape_edge'], lw=2.5))
    ax.annotate('', xy=end2, xytext=V,
               arrowprops=dict(arrowstyle='->', color=COLORS['shape_edge'], lw=2.5))
    
    if show_arc:
        angle1 = math.atan2(d1[1], d1[0])
        angle2 = math.atan2(d2[1], d2[0])
        
        arc_radius = ray_length * 0.25
        
        if angle1 > angle2:
            angle1, angle2 = angle2, angle1
        
        if angle2 - angle1 > math.pi:
            angle1, angle2 = angle2, angle1 + 2 * math.pi
        
        arc = Arc(V, 2*arc_radius, 2*arc_radius,
                 angle=0, theta1=math.degrees(angle1), theta2=math.degrees(angle2),
                 color=COLORS['angle_edge'], linewidth=2, zorder=3)
        ax.add_patch(arc)
        
        mid_angle = (angle1 + angle2) / 2
        label_pos = V + np.array([math.cos(mid_angle), math.sin(mid_angle)]) * arc_radius * 1.4
        ax.annotate(label, xy=label_pos, fontsize=14, fontweight='bold',
                   color=COLORS['angle_edge'], ha='center', va='center')
    
    if show_bisector:
        bisector_dir = d1 + d2
        bisector_dir = bisector_dir / np.linalg.norm(bisector_dir)
        bis_end = V + bisector_dir * ray_length * 0.8
        ax.plot([V[0], bis_end[0]], [V[1], bis_end[1]], '--',
               color=COLORS['auxiliary'], linewidth=1.5, zorder=3)
    
    _label_point(ax, V, 'O', offset=(-15, -15), use_greek=False)
    
    all_pts = np.array([V, end1, end2])
    margin = ray_length * 0.2
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def draw_parallel_lines(y1=1.8, y2=-1.8, x_range=(-4, 4), transversal_angle=60,
                        labels=('ε₁', 'ε₂', 'τ'), show_angles=True,
                        title='Παράλληλες ευθείες ε₁ // ε₂ με τέμνουσα τ'):
    """
    Draw two parallel lines intersected by a transversal (Chapter 4 - Παράλληλες ευθείες).
    Highlights alternating interior angles and corresponding angles.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.25, color=COLORS['grid'])
    
    # Parallel lines
    ax.plot(x_range, [y1, y1], color=COLORS['shape_edge'], lw=2.5, label=f'Ευθεία {labels[0]}')
    ax.plot(x_range, [y2, y2], color=COLORS['shape_edge'], lw=2.5, label=f'Ευθεία {labels[1]}')
    
    # Labels for parallel lines
    ax.text(x_range[1] + 0.3, y1, labels[0], fontsize=15, fontweight='bold', color=COLORS['label'], va='center')
    ax.text(x_range[1] + 0.3, y2, labels[1], fontsize=15, fontweight='bold', color=COLORS['label'], va='center')
    
    # Transversal line: passes through (0, 0) or offset
    theta_rad = math.radians(transversal_angle)
    # y - y0 = tan(theta) * (x - x0)
    slope = math.tan(theta_rad)
    
    # Points of intersection with y1 and y2
    x_int1 = y1 / slope
    x_int2 = y2 / slope
    
    # Extend transversal line
    ext = 1.5
    y_trans_top = y1 + ext
    y_trans_bot = y2 - ext
    x_trans_top = y_trans_top / slope
    x_trans_bot = y_trans_bot / slope
    
    ax.plot([x_trans_bot, x_trans_top], [y_trans_bot, y_trans_top], color=COLORS['auxiliary'], lw=2.5, label=f'Τέμνουσα {labels[2]}')
    ax.text(x_trans_top + 0.2, y_trans_top + 0.2, labels[2], fontsize=15, fontweight='bold', color=COLORS['auxiliary'])
    
    # Mark intersection points
    _label_point(ax, (x_int1, y1), 'A', offset=(-15, 10), use_greek=True, fontsize=13)
    _label_point(ax, (x_int2, y2), 'B', offset=(10, -15), use_greek=True, fontsize=13)
    
    if show_angles:
        # Alternating interior angles (εντός εναλλάξ)
        arc_r = 0.8
        # Angle at A (lower-left quadrant inside the parallel band)
        arc1 = Arc((x_int1, y1), 2*arc_r, 2*arc_r, angle=0, theta1=180, theta2=180 + transversal_angle,
                   color='#E91E63', lw=2)
        ax.add_patch(arc1)
        ax.text(x_int1 - 0.5, y1 - 0.4, 'ω₁', fontsize=12, fontweight='bold', color='#E91E63')
        
        # Angle at B (upper-right quadrant inside the parallel band)
        arc2 = Arc((x_int2, y2), 2*arc_r, 2*arc_r, angle=0, theta1=0, theta2=transversal_angle,
                   color='#E91E63', lw=2)
        ax.add_patch(arc2)
        ax.text(x_int2 + 0.5, y2 + 0.4, 'ω₂', fontsize=12, fontweight='bold', color='#E91E63')
        
        ax.text(0, (y1 + y2) / 2, 'ω₁ = ω₂ (Εντός εναλλάξ)', fontsize=13, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor='#FF9800'))
    
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        
    ax.set_xlim(min(x_range[0], x_trans_bot) - 1, max(x_range[1], x_trans_top) + 1.5)
    ax.set_ylim(y_trans_bot - 0.5, y_trans_top + 0.8)
    ax.axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def draw_3d_shape(shape_type='cylinder', title=None, dimensions=None):
    """
    Draw 3D solid figures for Greek High School Stereometry (Chapter 13 - Στερεά Σχήματα).
    Supported: cylinder, cone, pyramid, prism, cube, sphere.
    """
    st = shape_type.lower()
    dimensions = dimensions or {}

    def dim(name, default):
        value = float(dimensions.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive finite number')
        return value

    def number(value):
        return f'{value:g}'
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set_aspect('equal')
    ax.axis('off')
    
    blue = COLORS['shape_edge']
    gray = '#9E9E9E'
    orange = COLORS['auxiliary']
    green = COLORS['auxiliary2']
    
    if st == 'cylinder':
        if not title:
            title = 'Κύλινδρος (Cylinder)'
        radius_value = dim('radius', 2.0)
        height_value = dim('height', 5.0)
        r = 2.0
        h = float(np.clip(2 * height_value / radius_value, 2.5, 7.0))
        # Top ellipse
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(r * np.cos(theta), h/2 + 0.6 * np.sin(theta), color=blue, lw=2.5)
        # Bottom ellipse: front
        theta_f = np.linspace(np.pi, 2*np.pi, 100)
        ax.plot(r * np.cos(theta_f), -h/2 + 0.6 * np.sin(theta_f), color=blue, lw=2.5)
        # Bottom ellipse: back (hidden dashed)
        theta_b = np.linspace(0, np.pi, 100)
        ax.plot(r * np.cos(theta_b), -h/2 + 0.6 * np.sin(theta_b), '--', color=gray, lw=1.8)
        # Lateral edges
        ax.plot([-r, -r], [-h/2, h/2], color=blue, lw=2.5)
        ax.plot([r, r], [-h/2, h/2], color=blue, lw=2.5)
        # Height axis
        ax.plot([0, 0], [-h/2, h/2], '-.', color=orange, lw=1.8)
        ax.text(0.15, 0, f'υ = {number(height_value)}', fontsize=12, fontweight='bold', color=orange)
        # Radius line at top
        ax.plot([0, r], [h/2, h/2], '-', color=green, lw=2.0)
        ax.text(r/2, h/2 + 0.25, f'ρ = {number(radius_value)}', fontsize=12, fontweight='bold', color=green, ha='center')
        _label_point(ax, (0, h/2), 'O', offset=(-12, 5), use_greek=True)
        _label_point(ax, (0, -h/2), "O'", offset=(-15, -15), use_greek=True)

    elif st == 'cone':
        if not title:
            title = 'Κώνος (Cone)'
        radius_value = dim('radius', 2.0)
        height_value = dim('height', 5.0)
        r = 2.2
        h = float(np.clip(2.2 * height_value / radius_value, 2.8, 7.2))
        generator_value = math.hypot(radius_value, height_value)
        # Apex S
        S = (0, h/2)
        # Bottom ellipse: front
        theta_f = np.linspace(np.pi, 2*np.pi, 100)
        ax.plot(r * np.cos(theta_f), -h/2 + 0.6 * np.sin(theta_f), color=blue, lw=2.5)
        # Bottom ellipse: back (hidden dashed)
        theta_b = np.linspace(0, np.pi, 100)
        ax.plot(r * np.cos(theta_b), -h/2 + 0.6 * np.sin(theta_b), '--', color=gray, lw=1.8)
        # Slanted lateral edges (generatrices)
        ax.plot([-r, 0], [-h/2, h/2], color=blue, lw=2.5)
        ax.plot([r, 0], [-h/2, h/2], color=blue, lw=2.5)
        # Height SO
        ax.plot([0, 0], [-h/2, h/2], '-.', color=orange, lw=1.8)
        ax.text(-0.35, 0, f'υ = {number(height_value)}', fontsize=12, fontweight='bold', color=orange)
        # Radius line
        ax.plot([0, r], [-h/2, -h/2], '-', color=green, lw=2.0)
        ax.text(r/2, -h/2 - 0.35, f'ρ = {number(radius_value)}', fontsize=12, fontweight='bold', color=green, ha='center')
        # Generatrix label (γενέτειρα)
        ax.text(r/2 + 0.3, 0, f'λ = {number(generator_value)}', fontsize=12, fontweight='bold', color=blue)
        _label_point(ax, S, 'Κ', offset=(0, 10), use_greek=False, fontsize=13)
        _label_point(ax, (0, -h/2), 'O', offset=(-15, -12), use_greek=True, fontsize=12)
        
    elif st in ['pyramid', 'square_pyramid']:
        if not title:
            title = 'Κανονική Πυραμίδα (Regular Pyramid)'
        base_width = dim('base_width', 4.0)
        base_depth = dim('base_depth', base_width)
        height_value = dim('height', 5.0)
        width_scale = 4.0
        depth_scale = float(np.clip(base_depth / base_width, 0.35, 2.0))
        height_scale = float(np.clip(height_value / base_width, 0.6, 2.0))
        # Rectangular base in perspective
        A = np.array([-2.5, -1.8])
        B = np.array([1.5, -1.8])
        offset = np.array([1.2 * depth_scale, 1.0 * depth_scale])
        C = B + offset
        D = A + offset
        S = (A + B + C + D) / 4 + np.array([0, width_scale * height_scale])
        O = (A + B + C + D) / 4    # base center
        
        # Front base edges
        ax.plot([A[0], B[0]], [A[1], B[1]], color=blue, lw=2.5)
        ax.plot([B[0], C[0]], [B[1], C[1]], color=blue, lw=2.5)
        # Hidden base edges (dashed)
        ax.plot([A[0], D[0]], [A[1], D[1]], '--', color=gray, lw=1.8)
        ax.plot([C[0], D[0]], [C[1], D[1]], '--', color=gray, lw=1.8)
        # Front lateral edges
        ax.plot([S[0], A[0]], [S[1], A[1]], color=blue, lw=2.5)
        ax.plot([S[0], B[0]], [S[1], B[1]], color=blue, lw=2.5)
        ax.plot([S[0], C[0]], [S[1], C[1]], color=blue, lw=2.5)
        # Hidden lateral edge
        ax.plot([S[0], D[0]], [S[1], D[1]], '--', color=gray, lw=1.8)
        # Height SO
        ax.plot([S[0], O[0]], [S[1], O[1]], '-.', color=orange, lw=2.0)
        ax.text(O[0] + 0.15, (S[1] + O[1])/2, f'υ = {number(height_value)}', fontsize=12, fontweight='bold', color=orange)
        ax.text((A[0] + B[0])/2, A[1] - 0.35, f'α = {number(base_width)}', fontsize=11,
                fontweight='bold', color=green, ha='center')
        ax.text(C[0] + 0.15, (B[1] + C[1])/2, f'β = {number(base_depth)}', fontsize=11,
                fontweight='bold', color=green)

        # Labels
        _label_point(ax, S, 'Κ', offset=(0, 10), use_greek=False)
        _label_point(ax, A, 'A', offset=(-15, -15), use_greek=True)
        _label_point(ax, B, 'B', offset=(0, -18), use_greek=True)
        _label_point(ax, C, 'C', offset=(15, -5), use_greek=True)
        _label_point(ax, D, 'D', offset=(-15, 10), use_greek=True)
        _label_point(ax, O, 'O', offset=(8, -12), use_greek=True)
        
    elif st in ['prism', 'cube']:
        if st == 'cube':
            side = dim('side', 3.0)
            width_value = height_value = depth_value = side
            if not title:
                title = 'Κύβος (Cube)'
        else:
            width_value = dim('width', 4.0)
            height_value = dim('height', 3.0)
            depth_value = dim('depth', 2.0)
            if not title:
                title = 'Ορθογώνιο Παραλληλεπίπεδο (Prism / Cuboid)'
        visual_width = 3.8
        visual_height = float(np.clip(visual_width * height_value / width_value, 1.8, 5.0))
        visual_depth = float(np.clip(1.2 * depth_value / width_value * 2.0, 0.6, 2.2))
        # Front face
        A = np.array([-2.0, -2.0])
        B = np.array([-2.0 + visual_width, -2.0])
        C = np.array([B[0], -2.0 + visual_height])
        D = np.array([A[0], -2.0 + visual_height])
        # Back face offset
        dx, dy = visual_depth, visual_depth * 0.75
        A1 = A + np.array([dx, dy])
        B1 = B + np.array([dx, dy])
        C1 = C + np.array([dx, dy])
        D1 = D + np.array([dx, dy])
        
        # Front face (solid)
        for p1, p2 in [(A,B), (B,C), (C,D), (D,A)]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=blue, lw=2.5)
        # Top and right visible edges
        ax.plot([D[0], D1[0]], [D[1], D1[1]], color=blue, lw=2.5)
        ax.plot([C[0], C1[0]], [C[1], C1[1]], color=blue, lw=2.5)
        ax.plot([B[0], B1[0]], [B[1], B1[1]], color=blue, lw=2.5)
        ax.plot([D1[0], C1[0]], [D1[1], C1[1]], color=blue, lw=2.5)
        ax.plot([C1[0], B1[0]], [C1[1], B1[1]], color=blue, lw=2.5)
        # Hidden edges (dashed)
        ax.plot([A[0], A1[0]], [A[1], A1[1]], '--', color=gray, lw=1.8)
        ax.plot([A1[0], B1[0]], [A1[1], B1[1]], '--', color=gray, lw=1.8)
        ax.plot([A1[0], D1[0]], [A1[1], D1[1]], '--', color=gray, lw=1.8)
        
        # Labels
        # Textbook / 3D-viewer convention: bottom base ΑΒΓΔ, top base Α′Β′Γ′Δ′.
        # In this oblique drawing the bottom base is A, B, B1, A1 and the top
        # base is D, C, C1, D1.
        _label_point(ax, A, 'Α', offset=(-15, -15), use_greek=False)
        _label_point(ax, B, 'Β', offset=(10, -15), use_greek=False)
        _label_point(ax, B1, 'Γ', offset=(10, -4), use_greek=False)
        _label_point(ax, A1, 'Δ', offset=(-18, 6), use_greek=False)
        _label_point(ax, D, 'Α′', offset=(-20, 8), use_greek=False)
        _label_point(ax, C, 'Β′', offset=(10, 5), use_greek=False)
        _label_point(ax, C1, 'Γ′', offset=(10, 5), use_greek=False)
        _label_point(ax, D1, 'Δ′', offset=(-20, 8), use_greek=False)
        ax.text((A[0] + B[0])/2, A[1] - 0.35, f'α = {number(width_value)}', fontsize=11,
                fontweight='bold', color=green, ha='center')
        # A cube has a single edge length α; only the cuboid distinguishes γ (height).
        height_symbol = 'α' if st == 'cube' else 'γ'
        ax.text(B[0] + 0.15, (B[1] + C[1])/2, f'{height_symbol} = {number(height_value)}', fontsize=11,
                fontweight='bold', color=orange)
        if st == 'prism':
            # β on the bottom depth edge ΒΓ (B → B1), like the 3D viewer
            ax.text(B[0] + dx/2 + 0.2, B[1] + dy/2 - 0.3, f'β = {number(depth_value)}',
                    fontsize=11, fontweight='bold', color=green, ha='left')
        
    elif st == 'sphere':
        if not title:
            title = 'Σφαίρα (Sphere)'
        radius_value = dim('radius', 2.5)
        R = 2.5
        # Outer circle
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(R * np.cos(theta), R * np.sin(theta), color=blue, lw=2.5)
        # Equator front
        theta_f = np.linspace(np.pi, 2*np.pi, 150)
        ax.plot(R * np.cos(theta_f), 0.7 * np.sin(theta_f), color=blue, lw=2.0)
        # Equator back (dashed)
        theta_b = np.linspace(0, np.pi, 150)
        ax.plot(R * np.cos(theta_b), 0.7 * np.sin(theta_b), '--', color=gray, lw=1.8)
        # Center and Radius
        ax.plot([0, R * math.cos(math.radians(35))], [0, 0.7 * math.sin(math.radians(35))], '-', color=green, lw=2.2)
        ax.text(R * 0.45, 0.35, f'ρ = {number(radius_value)}', fontsize=12, fontweight='bold', color=green)
        _label_point(ax, (0, 0), 'O', offset=(-15, -5), use_greek=True)

    else:
        plt.close(fig)
        raise ValueError(f'Unsupported 3D shape: {shape_type}')
    
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf
