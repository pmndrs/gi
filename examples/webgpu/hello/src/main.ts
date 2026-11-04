// Placeholder: the smallest raw-WebGPU entry. It only asks the browser for a
// device; real use of @pmndrs/ddgi/core arrives with the first platform tasks.
import * as ddgiCore from '@pmndrs/ddgi/core';

const status = document.getElementById('status');

async function main(): Promise<string> {
    if (!navigator.gpu) return 'WebGPU is not available in this browser.';
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return 'No WebGPU adapter found.';
    // The host owns the device; the library will be handed this one.
    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (event) => console.error('WebGPU error', event.error));
    return `ddgi hello: device ready (core exports ${Object.keys(ddgiCore).length} names).`;
}

main().then((message) => {
    console.log(message);
    if (status) status.textContent = message;
});
