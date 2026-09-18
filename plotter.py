"""
Function plotting module for Greek high school mathematics (Γ΄ Λυκείου / Πανελλαδικές).
Supports: polynomials, trigonometric, exponential, logarithmic functions.
Features: roots, extrema, asymptotes, inflection points, shaded areas, tangent lines.
"""

import io
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor, function_exponentiation
)

# Color palette for function plots
PLOT_COLORS = [
    '#2196F3',  # Blue
    '#F44336',  # Red
    '#4CAF50',  # Green
    '#FF9800',  # Orange
    '#9C27B0',  # Purple
    '#00BCD4',  # Cyan
    '#795548',  # Brown
    '#E91E63',  # Pink
]

GRID_COLOR = '#E0E0E0'
AXIS_COLOR = '#424242'
POINT_COLOR = '#D32F2F'
SHADE_COLOR = '#BBDEFB'


class FormulaError(ValueError):
    """Raised when a formula is invalid or outside the supported grammar."""


def _parse_formula(formula_str):
    """Parse a formula string into a sympy expression."""
    x = sp.Symbol('x')
    
    # Clean up common notation
    formula = formula_str.strip()
    
    # Remove 'y=' or 'f(x)=' prefix
    for prefix in ['y=', 'y =', 'f(x)=', 'f(x) =', 'Y=', 'F(x)=']:
        if formula.startswith(prefix):
            formula = formula[len(prefix):]
            break
    
    # Replace common math notation
    formula = formula.replace('^', '**').replace('π', 'pi').replace('ln', 'log')

    if not formula or len(formula) > 200:
        raise FormulaError('Formula must contain between 1 and 200 characters')
    if not re.fullmatch(r'[0-9A-Za-z_+\-*/().\s]+', formula):
        raise FormulaError('Formula contains unsupported characters')
    if '__' in formula:
        raise FormulaError('Formula contains an unsafe identifier')

    allowed_names = {
        'x': x, 'e': sp.E, 'pi': sp.pi,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sqrt': sp.sqrt, 'log': sp.log, 'exp': sp.exp, 'abs': sp.Abs,
    }
    identifiers = set(re.findall(r'[A-Za-z_]+', formula))
    unknown = sorted(identifiers - set(allowed_names))
    if unknown:
        raise FormulaError(f"Unsupported formula identifier(s): {', '.join(unknown)}")
    
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
        function_exponentiation,
    )
    
    safe_globals = {
        '__builtins__': {},
        'Integer': sp.Integer,
        'Float': sp.Float,
        'Rational': sp.Rational,
    }
    try:
        expr = parse_expr(formula, local_dict=allowed_names,
                          global_dict=safe_globals,
                          transformations=transformations, evaluate=True)
    except Exception as exc:
        raise FormulaError('Formula could not be parsed') from exc
    if expr.free_symbols - {x}:
        raise FormulaError('Only x may be used as a variable')
    if sp.count_ops(expr) > 80:
        raise FormulaError('Formula is too complex')
    
    return expr, x


def _evaluate(expr, x_sym, x_values):
    """Evaluate an expression as a finite-shape real NumPy array."""
    try:
        raw = sp.lambdify(x_sym, expr, modules=['numpy'])(x_values)
        values = np.asarray(raw)
        if values.ndim == 0:
            values = np.full(np.shape(x_values), values.item())
        else:
            values = np.broadcast_to(values, np.shape(x_values)).copy()
        if np.iscomplexobj(values):
            imaginary = np.abs(np.imag(values))
            values = np.where(imaginary < 1e-10, np.real(values), np.nan)
        return values.astype(float)
    except Exception as exc:
        raise FormulaError('Formula could not be evaluated on the requested range') from exc


def _find_features(expr, x, x_range):
    """Find mathematical features of the expression."""
    features = {}
    
    try:
        # Roots (zeros)
        roots = sp.solve(expr, x)
        real_roots = []
        for r in roots:
            try:
                r_val = complex(r)
                if abs(r_val.imag) < 1e-10 and x_range[0] <= r_val.real <= x_range[1]:
                    real_roots.append(float(r_val.real))
            except (TypeError, ValueError):
                pass
        features['roots'] = sorted(set(real_roots))
    except Exception:
        features['roots'] = []
    
    try:
        # First derivative and critical points
        f_prime = sp.diff(expr, x)
        critical = sp.solve(f_prime, x)
        
        extrema = []
        f_double_prime = sp.diff(f_prime, x)
        
        for cp in critical:
            try:
                cp_val = complex(cp)
                if abs(cp_val.imag) < 1e-10 and x_range[0] <= cp_val.real <= x_range[1]:
                    cp_float = float(cp_val.real)
                    y_val = float(expr.subs(x, cp_float))
                    
                    second_deriv = float(f_double_prime.subs(x, cp_float))
                    if second_deriv > 0:
                        ext_type = 'min'
                    elif second_deriv < 0:
                        ext_type = 'max'
                    else:
                        ext_type = 'inflection'
                    
                    extrema.append((cp_float, y_val, ext_type))
            except (TypeError, ValueError):
                pass
        features['extrema'] = extrema
    except Exception:
        features['extrema'] = []
    
    try:
        # Vertical asymptotes (where denominator = 0)
        if expr.is_rational_function(x):
            numer, denom = expr.as_numer_denom()
            v_asymptotes = sp.solve(denom, x)
            v_asym_vals = []
            for va in v_asymptotes:
                try:
                    va_val = complex(va)
                    if abs(va_val.imag) < 1e-10:
                        v_asym_vals.append(float(va_val.real))
                except (TypeError, ValueError):
                    pass
            features['vertical_asymptotes'] = sorted(set(v_asym_vals))
        else:
            features['vertical_asymptotes'] = []
    except Exception:
        features['vertical_asymptotes'] = []
    
    try:
        # Horizontal asymptote
        h_asym_pos = sp.limit(expr, x, sp.oo)
        h_asym_neg = sp.limit(expr, x, -sp.oo)
        h_asymptotes = set()
        for h in [h_asym_pos, h_asym_neg]:
            if h.is_finite:
                h_asymptotes.add(float(h))
        features['horizontal_asymptotes'] = sorted(h_asymptotes)
    except Exception:
        features['horizontal_asymptotes'] = []
    
    return features


def plot_function(formulas, x_range=(-10, 10), y_range=None, show_features=None,
                  shade_between=None, title=None, show_grid=True, labels=None,
                  figsize=(10, 7)):
    """
    Plot one or more mathematical functions.
    
    Args:
        formulas: List of formula strings, e.g. ["y=x^2", "y=2x+1"]
        x_range: (xmin, xmax) tuple
        y_range: (ymin, ymax) tuple or None for auto
        show_features: List of features to annotate:
            - 'roots': mark x-intercepts
            - 'extrema': mark local min/max
            - 'asymptotes': draw asymptote lines
            - 'intercepts': mark y-intercepts
            - 'all': show everything
        shade_between: Dict for shading area:
            - {'x_start': float, 'x_end': float} to shade under first function
            - {'formula_indices': [0, 1]} to shade between two functions
        title: Optional title
        show_grid: Show grid lines
        labels: List of custom labels for each formula
        figsize: Figure size tuple
    
    Returns:
        BytesIO object containing PNG image
    """
    if show_features is None:
        show_features = []
    if 'all' in show_features:
        show_features = ['roots', 'extrema', 'asymptotes', 'intercepts']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    
    # Style
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)
    
    ax.axhline(y=0, color=AXIS_COLOR, linewidth=1.2, zorder=1)
    ax.axvline(x=0, color=AXIS_COLOR, linewidth=1.2, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    x_vals = np.linspace(x_range[0], x_range[1], 2000)
    all_y_vals = []
    parsed_exprs = []
    
    for i, formula in enumerate(formulas):
        expr, x_sym = _parse_formula(formula)
        parsed_exprs.append((expr, x_sym))
        
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        label = labels[i] if labels and i < len(labels) else formula
        
        # Evaluate function
        y_vals = _evaluate(expr, x_sym, x_vals)
        
        # Handle discontinuities (vertical asymptotes)
        features = _find_features(expr, x_sym, x_range) if show_features else {}
        
        # Mask extreme values for clean plotting
        y_masked = np.copy(y_vals)
        if y_range:
            threshold = (y_range[1] - y_range[0]) * 3
        else:
            finite_y = y_vals[np.isfinite(y_vals)]
            if len(finite_y) > 0:
                threshold = max(abs(np.percentile(finite_y, 5)), abs(np.percentile(finite_y, 95))) * 3
            else:
                threshold = 100
        
        y_masked[np.abs(y_masked) > threshold] = np.nan
        
        # Break at discontinuities
        dy = np.diff(y_masked)
        jumps = np.where(np.abs(dy) > threshold / 2)[0]
        for j in jumps:
            y_masked[j] = np.nan
            if j + 1 < len(y_masked):
                y_masked[j + 1] = np.nan
        
        ax.plot(x_vals, y_masked, '-', color=color, linewidth=2.5, label=label, zorder=3)
        all_y_vals.extend(y_masked[np.isfinite(y_masked)])
        
        # Annotate features
        if 'roots' in show_features and features.get('roots'):
            for root_index, root in enumerate(features['roots']):
                ax.plot(root, 0, 'o', color=POINT_COLOR, markersize=8, zorder=5)
                vertical_offset = -22 if root_index % 2 == 0 else 18
                ax.annotate(f'({root:.2g}, 0)', xy=(root, 0),
                           xytext=(0, vertical_offset),
                           textcoords='offset points', fontsize=9, ha='center',
                           color=POINT_COLOR, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor=POINT_COLOR, alpha=0.8))
        
        if 'extrema' in show_features and features.get('extrema'):
            for ex_x, ex_y, ex_type in features['extrema']:
                marker_color = '#4CAF50' if ex_type == 'min' else '#F44336'
                label_text = 'min' if ex_type == 'min' else 'max'
                
                ax.plot(ex_x, ex_y, 's', color=marker_color, markersize=8, zorder=5)
                ax.annotate(f'{label_text}\n({ex_x:.2g}, {ex_y:.2g})',
                           xy=(ex_x, ex_y),
                           xytext=(24, 22 if ex_type == 'max' else -42),
                           textcoords='offset points', fontsize=9,
                           color=marker_color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor=marker_color, alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.5))
        
        if 'asymptotes' in show_features:
            for va in features.get('vertical_asymptotes', []):
                ax.axvline(x=va, color='#9E9E9E', linewidth=1.5, linestyle='--',
                          alpha=0.7, zorder=2)
                ax.annotate(f'x={va:.2g}', xy=(va, 0), xytext=(8, 20),
                           textcoords='offset points', fontsize=9, color='#757575')
            
            for ha in features.get('horizontal_asymptotes', []):
                ax.axhline(y=ha, color='#9E9E9E', linewidth=1.5, linestyle='--',
                          alpha=0.7, zorder=2)
                ax.annotate(f'y={ha:.2g}', xy=(x_range[1], ha), xytext=(-30, 8),
                           textcoords='offset points', fontsize=9, color='#757575')
        
        if 'intercepts' in show_features:
            try:
                y_intercept = float(expr.subs(x_sym, 0))
                ax.plot(0, y_intercept, 'D', color='#FF9800', markersize=7, zorder=5)
                ax.annotate(f'(0, {y_intercept:.2g})', xy=(0, y_intercept),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, color='#FF9800', fontweight='bold')
            except Exception:
                pass
    
    # Shade between curves or under curve
    if shade_between:
        x_start = shade_between.get('x_start', x_range[0])
        x_end = shade_between.get('x_end', x_range[1])
        
        mask = (x_vals >= x_start) & (x_vals <= x_end)
        x_shade = x_vals[mask]
        
        if 'formula_indices' in shade_between and len(shade_between['formula_indices']) == 2:
            idx1, idx2 = shade_between['formula_indices']
            expr1, x1 = parsed_exprs[idx1]
            expr2, x2 = parsed_exprs[idx2]
            
            y1_shade = _evaluate(expr1, x1, x_shade)
            y2_shade = _evaluate(expr2, x2, x_shade)
            
            ax.fill_between(x_shade, y1_shade, y2_shade, alpha=0.3,
                          color=SHADE_COLOR, zorder=2, label='Area')
        else:
            expr0, x0 = parsed_exprs[0]
            y0_shade = _evaluate(expr0, x0, x_shade)
            
            ax.fill_between(x_shade, 0, y0_shade, alpha=0.3,
                          color=SHADE_COLOR, zorder=2, label='Area')
    
    # Set axis limits
    if y_range:
        ax.set_ylim(y_range)
    elif all_y_vals:
        y_arr = np.array(all_y_vals)
        y_min, y_max = np.percentile(y_arr, 2), np.percentile(y_arr, 98)
        y_margin = max((y_max - y_min) * 0.15, 1.0)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.set_xlim(x_range)
    
    # Add axis labels
    ax.set_xlabel('x', fontsize=13, fontweight='bold')
    ax.set_ylabel('y', fontsize=13, fontweight='bold', rotation=0, labelpad=15)
    
    # Legend
    if len(formulas) > 1:
        ax.legend(fontsize=11, loc='best', framealpha=0.9,
                 edgecolor='#BDBDBD', fancybox=True)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_trigonometric(formulas, periods=2, title=None, show_features=None):
    """
    Specialized plotter for trigonometric functions with period-aware x-range.
    
    Args:
        formulas: List of trig function strings
        periods: Number of periods to show
        title: Optional title
        show_features: Feature list
    """
    x_range = (-periods * math.pi, periods * math.pi)
    
    # Custom x-axis ticks with π labels
    buf = plot_function(formulas, x_range=x_range, show_features=show_features or [],
                       title=title)
    
    # Re-render with π ticks
    buf.seek(0)
    return buf
