import * as THREE from 'three';

export function createTitleSprite() {
    const canvas = document.createElement('canvas');
    canvas.width = 4096;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    // Use Times New Roman as requested
    ctx.font = '900 420px "Times New Roman", Times, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Drop shadow (first layer)
    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    ctx.shadowBlur = 60;
    ctx.shadowOffsetY = 15;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.fillText('WAVESENTINEL', 2048, 512);
    
    // Glow (second layer)
    ctx.shadowColor = 'rgba(56, 189, 248, 0.3)';
    ctx.shadowBlur = 30;
    ctx.shadowOffsetY = 0;
    ctx.fillText('WAVESENTINEL', 2048, 512);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    
    const material = new THREE.SpriteMaterial({ 
        map: texture, 
        transparent: true,
        fog: false, // Prevents scene fog from turning the sprite blue underwater
        depthWrite: false // Prevents sprite from messing with depth of transparent objects
    });
    
    const sprite = new THREE.Sprite(material);
    // Adjusted scale to fit nicely in the background (made larger)
    sprite.scale.set(400, 100, 1); 
    
    return sprite;
}
