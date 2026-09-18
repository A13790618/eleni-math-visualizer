"""
Eleni Math Visualization API — Custom Coze Plugin Backend
A Flask API that generates geometry figures and function graphs
for Greek high school mathematics education.

Endpoints:
    POST /api/geometry      — Draw geometric shapes (triangle, circle, etc.)
    POST /api/plot           — Plot mathematical functions
    GET  /api/health         — Health check
    GET  /images/<filename>  — Serve generated images
"""

import os
import json
import uuid
import time
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from geometry import (
    draw_triangle, draw_circle, draw_quadrilateral, draw_angle,
    draw_parallel_lines, draw_3d_shape
)
from plotter import FormulaError, plot_function
from validation import ValidationError, validate_geometry_request, validate_plot_request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
CORS(app)

# --- Image storage ---
IMAGE_DIR = Path('/tmp/eleni_images')
IMAGE_DIR.mkdir(exist_ok=True)

# Clean temporary generated artifacts every 30 minutes.
IMAGE_TTL_SECONDS = 86400

def _cleanup_old_artifacts():
    """Remove generated image previews after their TTL."""
    while True:
        time.sleep(1800)
        now = time.time()
        try:
            for artifact in IMAGE_DIR.iterdir():
                if (artifact.is_file()
                        and (now - artifact.stat().st_mtime) > IMAGE_TTL_SECONDS):
                    artifact.unlink()
        except OSError:
            app.logger.exception('Temporary artifact cleanup failed')

cleanup_thread = threading.Thread(target=_cleanup_old_artifacts, daemon=True)
cleanup_thread.start()


def _save_image(buf):
    """Save image buffer to file and return the filename."""
    filename = f"{uuid.uuid4().hex}.png"
    filepath = IMAGE_DIR / filename
    with open(filepath, 'wb') as f:
        f.write(buf.read())
    return filename


def _public_url(path):
    """Generate an absolute public URL for an application path."""
    if 'BASE_URL' in os.environ:
        base_url = os.environ['BASE_URL'].rstrip('/')
    else:
        scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
        host = request.headers.get('X-Forwarded-Host', request.host)
        base_url = f"{scheme}://{host}"
    return f"{base_url}{path}"


def _image_url(filename):
    return _public_url(f'/images/{filename}')


def _viewer_serializer():
    """Create a stable signer; production should use API_KEY or VIEWER_SIGNING_KEY."""
    secret = (os.environ.get('VIEWER_SIGNING_KEY')
              or os.environ.get('API_KEY')
              or 'eleni-local-development-only')
    return URLSafeTimedSerializer(secret, salt='eleni-interactive-viewer-v1')


def _create_viewer_token(shape_type, dimensions, title):
    """Encode a validated solid configuration in a tamper-resistant URL token."""
    payload = {
        'shape': shape_type,
        'dimensions': dimensions or {},
        'title': title,
    }
    return _viewer_serializer().dumps(payload)


@app.before_request
def require_optional_api_key():
    """Protect generation endpoints when API_KEY is configured by the owner."""
    expected = os.environ.get('API_KEY')
    if not expected or request.path == '/api/health' or not request.path.startswith('/api/'):
        return None
    supplied = request.headers.get('X-API-Key')
    authorization = request.headers.get('Authorization', '')
    if authorization.startswith('Bearer '):
        supplied = authorization[7:]
    if supplied != expected:
        return jsonify({
            'success': False,
            'error': 'Unauthorized',
            'message': 'A valid API key is required'
        }), 401
    return None


# ==================== ROUTES ====================

@app.route('/', methods=['GET'])
def index():
    """Small public landing response for the Hugging Face Space."""
    return jsonify({
        'status': 'ok',
        'service': 'Eleni Math Visualization API',
        'version': '1.1.0',
        'documentation': '/openapi.json',
        'health': '/api/health',
        'capabilities': ['2d_geometry', 'interactive_3d', 'function_plotting']
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'Eleni Math Visualization API',
        'version': '1.1.0',
        'capabilities': ['2d_geometry', 'interactive_3d', 'function_plotting']
    })


@app.route('/openapi.json', methods=['GET'])
def serve_openapi():
    """Serve the OpenAPI 3.0 specification for Coze."""
    openapi_path = Path(__file__).parent / 'openapi.json'
    if openapi_path.exists():
        with open(openapi_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'openapi.json not found'}), 404


@app.route('/api/geometry', methods=['POST'])
def api_geometry():
    """
    Draw geometric shapes.
    
    Request body:
    {
        "shape": "triangle" | "circle" | "quadrilateral" | "angle",
        "vertices": [[x1,y1], [x2,y2], ...],     // for triangle/quad
        "center": [x, y],                          // for circle
        "radius": float,                            // for circle
        "labels": ["A", "B", "C"],                  // vertex labels
        "features": ["altitude_from_A", "median_from_B", ...],
        "title": "Optional title",
        "use_greek": true,                          // convert labels to Greek
        "show_grid": true,
        "show_axis": true
    }
    """
    try:
        data = validate_geometry_request(request.get_json(silent=True))
        
        shape = data.get('shape', 'triangle').lower()
        title = data.get('title')
        use_greek = data.get('use_greek', True)
        show_grid = data.get('show_grid', True)
        show_axis = data.get('show_axis', True)
        labels = data.get('labels')
        features = data.get('features', [])
        interactive_model = None
        
        if shape == 'triangle':
            vertices = data.get('vertices')
            
            buf = draw_triangle(
                vertices=vertices,
                labels=labels,
                features=features,
                title=title,
                show_grid=show_grid,
                show_axis=show_axis,
                use_greek=use_greek
            )
        
        elif shape == 'circle':
            center = data.get('center', [0, 0])
            radius = data.get('radius', 2)
            circle_labels = data.get('labels', {})
            if isinstance(circle_labels, list):
                circle_labels = {'center': circle_labels[0] if circle_labels else 'O'}
            
            buf = draw_circle(
                center=center,
                radius=radius,
                labels=circle_labels,
                features=features,
                point_angles=data.get('point_angles'),
                point_labels=data.get('point_labels'),
                title=title,
                show_grid=show_grid,
                show_axis=show_axis,
                use_greek=use_greek
            )
        
        elif shape in ['quadrilateral', 'square', 'rectangle', 'parallelogram',
                       'rhombus', 'trapezoid']:
            vertices = data.get('vertices')
            
            buf = draw_quadrilateral(
                vertices=vertices,
                shape_type=shape,
                labels=labels,
                features=features,
                title=title,
                show_grid=show_grid,
                show_axis=show_axis,
                use_greek=use_greek
            )
        
        elif shape == 'angle':
            vertex = data.get('vertex', [0, 0])
            ray1 = data.get('ray1_end', [3, 0])
            ray2 = data.get('ray2_end', [2, 2.5])
            angle_label = data.get('angle_label', 'α')
            show_bisector = data.get('show_bisector', False)
            
            buf = draw_angle(
                vertex=vertex,
                ray1_end=ray1,
                ray2_end=ray2,
                label=angle_label,
                show_bisector=show_bisector,
                title=title,
                show_grid=show_grid,
                show_axis=show_axis
            )
        
        elif shape in ['parallel_lines', 'parallel']:
            transversal_angle = data.get('transversal_angle', 60)
            buf = draw_parallel_lines(
                transversal_angle=transversal_angle,
                title=title or 'Παράλληλες ευθείες ε₁ // ε₂ με τέμνουσα τ'
            )
        
        elif shape in ['cylinder', 'cone', 'pyramid', 'square_pyramid',
                       'prism', 'cube', 'sphere', '3d_shape']:
            shape_type = data.get('shape_type', shape)
            buf = draw_3d_shape(
                shape_type=shape_type,
                title=title,
                dimensions=data.get('dimensions')
            )
            interactive_model = {
                'shape_type': shape_type,
                'dimensions': data.get('dimensions', {}),
                'title': title,
            }
        
        else:
            return jsonify({'error': f'Unknown shape: {shape}. '
                          'Supported: triangle, circle, quadrilateral, square, '
                          'rectangle, parallelogram, angle, parallel_lines, '
                          'cylinder, cone, pyramid, prism, sphere'}), 400
        
        filename = _save_image(buf)
        image_url = _image_url(filename)
        
        response = {
            'success': True,
            'image_url': image_url,
            'shape': shape,
            'message': f'Successfully drew {shape}'
        }
        if interactive_model:
            viewer_token = _create_viewer_token(**interactive_model)
            response.update({
                'interactive': True,
                'viewer_url': _public_url(f'/viewer/{viewer_token}')
            })
        return jsonify(response)
    
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e),
                        'message': 'Invalid geometry request'}), 400
    except Exception:
        app.logger.exception('Geometry generation failed')
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to generate geometry image'
        }), 500


@app.route('/api/plot', methods=['POST'])
def api_plot():
    """
    Plot mathematical functions.
    
    Request body:
    {
        "formulas": ["y=x^2-3x+2", "y=2x+1"],
        "x_range": [-10, 10],
        "y_range": [-5, 15],               // optional
        "show_features": ["roots", "extrema", "asymptotes", "intercepts", "all"],
        "shade": {                           // optional
            "x_start": 1,
            "x_end": 3
        },
        "title": "Optional title",
        "labels": ["f(x) = x²-3x+2"],      // optional custom labels
        "show_grid": true
    }
    """
    try:
        data = validate_plot_request(request.get_json(silent=True))
        formulas = data['formulas']
        
        x_range = tuple(data.get('x_range', [-10, 10]))
        y_range = data.get('y_range')
        if y_range:
            y_range = tuple(y_range)
        
        show_features = data.get('show_features', [])
        title = data.get('title')
        labels = data.get('labels')
        show_grid = data.get('show_grid', True)
        
        shade_between = data.get('shade')
        
        buf = plot_function(
            formulas=formulas,
            x_range=x_range,
            y_range=y_range,
            show_features=show_features,
            shade_between=shade_between,
            title=title,
            show_grid=show_grid,
            labels=labels
        )
        
        filename = _save_image(buf)
        image_url = _image_url(filename)
        
        return jsonify({
            'success': True,
            'image_url': image_url,
            'formulas': formulas,
            'message': f'Successfully plotted {len(formulas)} function(s)'
        })
    
    except (ValidationError, FormulaError) as e:
        return jsonify({'success': False, 'error': str(e),
                        'message': 'Invalid plot request'}), 400
    except Exception:
        app.logger.exception('Function plotting failed')
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to generate function plot'
        }), 500


@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve generated images."""
    response = send_from_directory(IMAGE_DIR, filename, mimetype='image/png')
    response.headers['Cache-Control'] = f'public, max-age={IMAGE_TTL_SECONDS}, immutable'
    return response


@app.route('/viewer/<viewer_token>', methods=['GET'])
def interactive_solid_viewer(viewer_token):
    """Render a self-contained Three.js viewer for a generated solid."""
    if len(viewer_token) > 2048:
        abort(404)
    try:
        model = _viewer_serializer().loads(viewer_token, max_age=604800)
    except (BadSignature, SignatureExpired):
        abort(404)
    response = app.make_response(render_template('solid_viewer.html', model=model))
    response.headers['Cache-Control'] = 'private, max-age=3600'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist.',
        'available_endpoints': [
            'POST /api/geometry',
            'POST /api/plot',
            'GET /api/health',
            'GET /images/<filename>',
            'GET /viewer/<model_id>'
        ]
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
