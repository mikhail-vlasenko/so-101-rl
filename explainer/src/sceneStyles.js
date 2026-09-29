import * as THREE from 'three';

/** Core material treatments. Additional set geometry lives in sceneEnvironments. */
export const SCENE_STYLES = {
  studio: { title: '01 / Dark studio', description: 'Lit materials · dark stage', background: '#080d15', floor: '#17212c', grid: '#2c3c4c', text: '#aabacb', surfaceText: '#b7f0b1', positionText: '#a2edf5', eye: [43, 39, 64], target: [5, 7, 20] },
  porcelain: { title: '02 / Tabletop exhibit', description: 'Cutaway plinth · ceramic rig · physical scale', background: '#cbc7be', floor: '#b5b3a9', grid: '#bab7ae', text: '#4f5865', surfaceText: '#27774c', positionText: '#146d89', eye: [57, 57, 76], target: [5, 2, 20] },
  blueprint: { title: '03 / Optical space', description: 'Image planes · view volumes · projected masks', background: '#050914', floor: '#080e21', grid: '#24467a', text: '#9db9e9', surfaceText: '#9aefac', positionText: '#8decff', eye: [58, 35, 52], target: [6, 9, 19] },
};

export function applySceneStyle(scene, style) {
  if (style === 'studio') return;
  const theme = SCENE_STYLES[style];
  const outlines = [];
  scene.traverse((object) => {
    if (object instanceof THREE.GridHelper) {
      object.visible = style === 'blueprint';
      object.material.opacity = 0.12;
    }
    if (object instanceof THREE.HemisphereLight) {
      object.color.set(style === 'porcelain' ? '#fff7ec' : '#a6bfff');
      object.groundColor.set(style === 'porcelain' ? '#b3ac9f' : '#080c20');
      object.intensity = style === 'porcelain' ? 2.2 : 1;
    }
    if (object instanceof THREE.DirectionalLight) {
      object.color.set(style === 'porcelain' ? '#fff4e1' : '#597ed0');
      object.intensity = style === 'porcelain' ? 2 : 1;
      if (object.castShadow) object.shadow.radius = 6;
    }
    if (object instanceof THREE.Line && !(object instanceof THREE.GridHelper)) {
      object.material.color.set(style === 'porcelain' ? '#68747b' : '#507ebd');
      object.material.opacity = Math.max(object.material.opacity, 0.4);
    }
    if (!(object instanceof THREE.Mesh) || object.geometry instanceof THREE.PlaneGeometry) return;
    if (object.material instanceof THREE.MeshBasicMaterial) {
      if (style === 'porcelain') {
        if (object instanceof THREE.InstancedMesh) object.material.color.set('#27774c');
        else object.material.color.set('#147f9a');
      }
      return;
    }
    if (style === 'porcelain') {
      object.material.color.set(object.name === 'sponge' ? '#d38a51' : '#a8acc1');
      object.material.roughness = 0.94;
      object.material.metalness = 0;
      return;
    }
    // Opaque dark faces retain occlusion; extracted edges avoid triangle clutter.
    object.material.dispose();
    object.material = new THREE.MeshBasicMaterial({ color: '#0b152b', toneMapped: false });
    object.castShadow = false;
    let edgeSource = object.geometry;
    // Bevel segments have shallow normal changes; use the box's principal
    // edges so this treatment reads as a drawing rather than a tessellated mesh.
    if (object.geometry instanceof THREE.BoxGeometry) {
      object.geometry.computeBoundingBox();
      const size = object.geometry.boundingBox.getSize(new THREE.Vector3());
      edgeSource = new THREE.BoxGeometry(size.x, size.y, size.z);
    }
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(edgeSource, 12),
      new THREE.LineBasicMaterial({ color: object.name === 'sponge' ? '#f1b567' : '#638ac6', toneMapped: false }),
    );
    if (edgeSource !== object.geometry) edgeSource.dispose();
    outlines.push([object, edges]);
  });
  for (const [object, edges] of outlines) object.add(edges);
  scene.background.set(theme.background);
}
