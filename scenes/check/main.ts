import { runCheckPage } from '../src/check-page';

const canvas = document.getElementById('canvas');
const status = document.getElementById('status');
if (!(canvas instanceof HTMLCanvasElement) || !status) throw new Error('check/index.html is missing #canvas or #status');
void runCheckPage(canvas, status);
