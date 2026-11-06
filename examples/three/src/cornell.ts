/**
 * The scene of three r187's `webgpu_lightprobes.html`, copied as it is: a
 * 6 × 5 × 6 m Cornell box with two boxes, a sphere and a point light. Both
 * examples use it unchanged, so the only difference from three's example
 * is the GI: `GIVolume` instead of `LightProbeGrid`.
 */

import * as THREE from 'three/webgpu';

export interface CornellScene {
    readonly scene: THREE.Scene;
    readonly camera: THREE.PerspectiveCamera;
    readonly light: THREE.PointLight;
}

export function createCornellScene(aspect: number): CornellScene {
    const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
    camera.position.set(0, 2.5, 8);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xcccccc });
    const redMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const greenMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });

    const wall = (width: number, height: number, material: THREE.Material, position: [number, number, number], rotation: [number, number]) => {
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
        mesh.rotation.x = rotation[0];
        mesh.rotation.y = rotation[1];
        mesh.position.set(...position);
        mesh.receiveShadow = true;
        scene.add(mesh);
    };
    wall(6, 6, wallMaterial, [0, 0, 0], [-Math.PI / 2, 0]); // floor
    wall(6, 6, wallMaterial, [0, 5, 0], [Math.PI / 2, 0]); // ceiling
    wall(6, 5, wallMaterial, [0, 2.5, -3], [0, 0]); // back
    wall(6, 5, wallMaterial, [0, 2.5, 3], [0, Math.PI]); // front
    wall(6, 5, redMaterial, [-3, 2.5, 0], [0, Math.PI / 2]); // left, red
    wall(6, 5, greenMaterial, [3, 2.5, 0], [0, -Math.PI / 2]); // right, green

    const objectMaterial = new THREE.MeshStandardMaterial({ color: 0xeeeeee });
    const object = (geometry: THREE.BufferGeometry, material: THREE.Material, position: [number, number, number], rotationY: number) => {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...position);
        mesh.rotation.y = rotationY;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
    };
    object(new THREE.BoxGeometry(1.2, 2.5, 1.2), objectMaterial, [-0.8, 1.25, -0.8], Math.PI / 8); // tall box
    object(new THREE.BoxGeometry(1.2, 1.2, 1.2), objectMaterial, [1, 0.6, 0.5], -Math.PI / 6); // short box
    object(new THREE.SphereGeometry(0.5, 32, 32), objectMaterial.clone(), [1, 1.9, 0.5], 0); // sphere

    const light = new THREE.PointLight(0xffffff, 20);
    light.position.set(0, 4.5, 0);
    light.castShadow = true;
    light.shadow.mapSize.setScalar(256);
    light.shadow.radius = 10;
    light.shadow.normalBias = -0.02;
    scene.add(light);

    return { scene, camera, light };
}
