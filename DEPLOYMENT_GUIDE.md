# Deployment and Coze Connection Guide

## Active Production Deployment (Render)

The live production backend is currently deployed and running on **Render**:
- **Service Name**: `eleni-math-visualizer`
- **Base URL**: `https://eleni-math-visualizer.onrender.com`
- **OpenAPI 3.0 Specification**: `https://eleni-math-visualizer.onrender.com/openapi.json`
- **GitHub Repository**: `https://github.com/A13790618/eleni-math-visualizer`
- **Auth Header**: `X-API-Key`
- **API Key**: `4fb33ce92de9f1289c86ac01f9020c95`
- **Viewer Signing Key**: `19a756992b5ddf951a3e5a01bee90e3d`

> ⚠️ **Note on Render Free Tier**: On the free tier, the web service spins down after 15 minutes of inactivity and takes ~30–50 seconds to wake up upon receiving an incoming request. For production classroom usage with Coze without cold-start timeouts, upgrading to the $7/month Render Starter instance is recommended.

---

## Alternative / Backup Deployment (Hugging Face Spaces)

1. Sign in to the client's Hugging Face account.
2. Create a new Space with the **Docker** SDK.
3. Upload the application files listed in the root `Dockerfile`.
4. In Space settings, create a secret named `API_KEY` with a long random value.
5. Wait for the Space status to become **Running**.
6. Open `https://<space-host>/api/health` and confirm that `status` is `ok`.
7. Replace the placeholder server URL in `openapi.json` with the Space hostname.

## 3. Register the plugin in Coze

1. In the Eleni workspace, open **Personal / Plugins / Create plugin**.
2. Choose import from an OpenAPI URL.
3. Use `https://<space-host>/openapi.json`.
4. Configure authentication with the same secret:
   - header name: `X-API-Key`
   - value: the Space `API_KEY` secret
5. Enable the `drawGeometry` and `plotFunctions` operations.
6. Run each operation once from Coze's debug panel before adding it to the bot.

## 4. Bind the result to a Coze Card

Create an image result Card and bind:

- Image component source → `{{image_url}}` (displays the inline 2D diagram or 3D solid preview)
- Button component target / URL → `{{viewer_url}}` (prominently labeled "Προβολή 3D / Rotate 3D" — required for actual student rotation, zoom, and touch controls)
- Title or caption text → `{{message}}`

Attach the Card to the plugin operations. The image source must use the tool's
direct `image_url` and the interactive button must link to `viewer_url`.
For 2D geometry and function plots where `viewer_url` is omitted, the Card gracefully displays only the preview image.

## 5. Acceptance checks

Verify these prompts in the actual student bot:

1. Triangle ΑΒΓ with an altitude and right-angle marker.
2. Circle with a tangent and perpendicular radius.
3. Circle with chord ΑΒ and an inscribed angle at Γ.
4. Cylinder with radius 2 and height 7 — verify preview image in chat and click button to open rotatable 3D model.
5. Cone with radius 3 and height 4 — rotate and zoom in 3D viewer.
6. Square pyramid with base side 4 and height 6 — rotate in 3D viewer.
7. Cube with side 5 and rectangular prism with unequal dimensions.
8. Sphere with a labelled radius.
9. Plot `f(x)=x^2-3x+2` with roots and minimum.
10. Plot two functions and shade the area between them.

For every check, confirm mathematical correctness, readable Greek labels, an
inline image Card, a functional one-click 3D viewer link for solids, and a response time suitable for the classroom.
