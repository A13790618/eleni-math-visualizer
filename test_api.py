#!/usr/bin/env python3
"""
Test script for the Eleni Math Visualization API.
Runs the server locally and tests all endpoints.
"""

import os
import sys
import json
import time
import subprocess
import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:5000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")

SESSION = requests.Session()
if API_KEY:
    SESSION.headers.update({"X-API-Key": API_KEY})


def wait_for_server(timeout=20):
    """Wait for the server to start."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = SESSION.get(f"{BASE_URL}/api/health", timeout=5)
            if r.status_code == 200:
                print("✅ Server is running!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)
    print("❌ Server failed to respond")
    return False


def test_health():
    """Test health endpoint."""
    r = SESSION.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    print("✅ Health check passed")


def test_triangle():
    """Test triangle drawing."""
    r = SESSION.post(f"{BASE_URL}/api/geometry", json={
        "shape": "triangle",
        "vertices": [[0, 0], [4, 0], [2, 3]],
        "labels": ["A", "B", "C"],
        "features": ["altitude_from_A"],
        "title": "Τρίγωνο ΑΒΓ με ύψος",
        "use_greek": True
    })
    assert r.status_code == 200
    data = r.json()
    assert data['success'] == True
    print(f"✅ Triangle: {data['image_url']}")
    
    # Verify image is accessible
    img_r = SESSION.get(data['image_url'])
    assert img_r.status_code == 200
    assert len(img_r.content) > 1000  # Should be a real image
    print(f"   Image size: {len(img_r.content)} bytes")
    
    # Save locally for inspection
    with open('/tmp/test_triangle.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_triangle.png")


def test_triangle_all_features():
    """Test triangle with all features."""
    r = SESSION.post(f"{BASE_URL}/api/geometry", json={
        "shape": "triangle",
        "vertices": [[0, 0], [5, 0], [2, 4]],
        "labels": ["A", "B", "C"],
        "features": ["all_altitudes", "circumscribed_circle"],
        "title": "Τρίγωνο με ύψη και περιγεγραμμένο κύκλο"
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Triangle (all features): {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_triangle_full.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_triangle_full.png")


def test_circle():
    """Test circle drawing."""
    r = SESSION.post(f"{BASE_URL}/api/geometry", json={
        "shape": "circle",
        "center": [0, 0],
        "radius": 3,
        "features": ["radius_line", "tangent"],
        "title": "Κύκλος με ακτίνα και εφαπτομένη"
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Circle: {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_circle.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_circle.png")


def test_square():
    """Test square drawing."""
    r = SESSION.post(f"{BASE_URL}/api/geometry", json={
        "shape": "square",
        "vertices": [[0, 0], [3, 0], [3, 3], [0, 3]],
        "labels": ["A", "B", "C", "D"],
        "features": ["diagonals", "right_angles"],
        "title": "Τετράγωνο ΑΒΓΔ"
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Square: {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_square.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_square.png")


def test_parabola():
    """Test parabola plotting."""
    r = SESSION.post(f"{BASE_URL}/api/plot", json={
        "formulas": ["y=x**2-3*x+2"],
        "x_range": [-2, 5],
        "show_features": ["roots", "extrema", "intercepts"],
        "title": "f(x) = x² - 3x + 2"
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Parabola: {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_parabola.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_parabola.png")


def test_trig():
    """Test trigonometric function."""
    r = SESSION.post(f"{BASE_URL}/api/plot", json={
        "formulas": ["y=sin(x)", "y=cos(x)"],
        "x_range": [-6.28, 6.28],
        "y_range": [-1.5, 1.5],
        "show_features": ["roots"],
        "title": "sin(x) και cos(x)",
        "labels": ["f(x) = sin(x)", "g(x) = cos(x)"]
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Trig functions: {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_trig.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_trig.png")


def test_rational():
    """Test rational function with asymptotes."""
    r = SESSION.post(f"{BASE_URL}/api/plot", json={
        "formulas": ["y=1/(x-2)"],
        "x_range": [-5, 8],
        "show_features": ["asymptotes", "intercepts"],
        "title": "f(x) = 1/(x-2)"
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Rational function: {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_rational.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_rational.png")


def test_shaded_area():
    """Test area under curve."""
    r = SESSION.post(f"{BASE_URL}/api/plot", json={
        "formulas": ["y=x**2"],
        "x_range": [-3, 3],
        "show_features": ["roots", "extrema"],
        "shade": {"x_start": 0, "x_end": 2},
        "title": "Εμβαδόν κάτω από f(x) = x²"
    })
    data = r.json()
    assert data['success'] == True
    print(f"✅ Shaded area: {data['image_url']}")
    
    img_r = SESSION.get(data['image_url'])
    with open('/tmp/test_shaded.png', 'wb') as f:
        f.write(img_r.content)
    print("   Saved to /tmp/test_shaded.png")


def test_interactive_solid():
    """Test interactive 3D solid geometry and viewer URL."""
    r = SESSION.post(f"{BASE_URL}/api/geometry", json={
        "shape": "cylinder",
        "dimensions": {"radius": 2, "height": 7},
        "title": "Κύλινδρος — r=2, h=7"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["interactive"] is True
    assert "viewer_url" in data
    viewer_url = data["viewer_url"]
    print(f"✅ Interactive Cylinder: {viewer_url}")
    
    # Verify viewer_url returns 200 and valid HTML with OrbitControls and readiness marker
    vr = SESSION.get(viewer_url)
    assert vr.status_code == 200
    assert "OrbitControls" in vr.text
    assert "__ELENI_VIEWER_READY__" in vr.text
    print("   Viewer page verified with 3D OrbitControls & model payload")


def main():
    print("=" * 60)
    print("🧪 Testing Eleni Math Visualization API")
    print("=" * 60)
    
    if not wait_for_server():
        print("\n❌ Could not connect to server. Make sure it's running:")
        print("   source venv/bin/activate && python app.py")
        sys.exit(1)
    
    tests = [
        test_health,
        test_triangle,
        test_triangle_all_features,
        test_circle,
        test_square,
        test_parabola,
        test_trig,
        test_rational,
        test_shaded_area,
        test_interactive_solid,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! Check /tmp/test_*.png for output images.")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
