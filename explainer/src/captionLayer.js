import * as THREE from 'three';
import { FILM_CAPTIONS, ease } from './filmTimeline.js';

const TEXT_HEIGHT = 96;
const ROW_STEP = 76;
const MAIN_TOP = 555;

/** Rasterize each phrase once at native resolution. Position, color, and
 * opacity animate on GPU quads instead of repainting/uploading a full-screen
 * canvas every frame. The same layer remains part of PNG/video exports.
 */
export function buildCaptionLayer(scene) {
  const planes = Array.from({ length: 3 }, (_, index) => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, TEXT_HEIGHT * 2 / 720),
      new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false, toneMapped: false }));
    mesh.visible = false;
    mesh.renderOrder = 3 + index;
    scene.add(mesh);
    return mesh;
  });
  const textures = new Map();
  let width = 0;
  let height = 0;
  let rasterizations = 0;
  return {
    resize(drawingWidth, drawingHeight) {
      const textHeight = Math.round(drawingHeight * TEXT_HEIGHT / 720);
      if (drawingWidth === width && textHeight === height) return;
      width = drawingWidth;
      height = textHeight;
      for (const texture of textures.values()) texture.dispose();
      textures.clear();
      for (const { text } of FILM_CAPTIONS) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(width / 1280, 0, 0, canvas.height / TEXT_HEIGHT, 0, 0);
        ctx.textAlign = 'center';
        ctx.font = '400 28px Arial';
        ctx.fillStyle = '#ffffff';
        const lines = text.split('\n');
        const baseline = lines.length === 1 ? 49 : 31;
        lines.forEach((line, index) => ctx.fillText(line, 640, baseline + index * 37));
        const texture = new THREE.CanvasTexture(canvas);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.generateMipmaps = false;
        texture.minFilter = THREE.LinearFilter;
        textures.set(text, texture);
        rasterizations++;
      }
    },
    update(state) {
      planes.forEach((mesh, index) => {
        const item = state.captionStack[index];
        mesh.visible = Boolean(item);
        if (!item) return;
        const texture = textures.get(item.text);
        if (!texture) throw new Error(`Missing cached caption: ${item.text}`);
        if (mesh.material.map !== texture) {
          mesh.material.map = texture;
          mesh.material.needsUpdate = true;
        }
        mesh.position.y = 1 - (MAIN_TOP + TEXT_HEIGHT / 2 + item.slot * ROW_STEP) * 2 / 720;
        mesh.material.color.setRGB((134 + 106 * item.emphasis) / 255, (147 + 98 * item.emphasis) / 255, (159 + 90 * item.emphasis) / 255, THREE.SRGBColorSpace);
        const lines = item.text.split('\n');
        const baseline = MAIN_TOP + (lines.length === 1 ? 49 : 31) + item.slot * ROW_STEP;
        const textBottom = baseline + (lines.length - 1) * 37 + 6;
        const edgeOpacity = ease(baseline - 28, 480, 520) * (1 - ease(textBottom, 710, 720));
        mesh.material.opacity = state.captionOpacity * item.opacity * edgeOpacity;
      });
    },
    stats() {
      return { width, height, textures: textures.size, rasterizations };
    },
    dispose() {
      for (const texture of textures.values()) texture.dispose();
      textures.clear();
      for (const mesh of planes) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
    },
  };
}
