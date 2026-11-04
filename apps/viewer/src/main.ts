// Placeholder entry. The real viewer shell arrives in T0.5.
import * as ddgiCore from '@pmndrs/ddgi/core';

const status = document.getElementById('status');
if (status) {
    status.textContent = `ddgi viewer: workspace is wired (core exports ${Object.keys(ddgiCore).length} names).`;
}
