/**
 * Where the runner's server (run.ts) takes each view's GI term from the page
 * (main.ts). Its own file because run.ts runs in Node and must not load the
 * page's browser code.
 */
export const UPLOAD_PATH = '/m4-textured-accuracy/upload';
