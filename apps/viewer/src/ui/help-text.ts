/**
 * Hover help for every control, in plain language: what it does and what you
 * will see. Kept in one file so the wording can be read and fixed in one go.
 */

export const HELP = {
    scene:
        'Pick a test scene. Each one opens on a lit view. Big ones (Sponza, Bistro) take a few seconds to load; a progress bar shows how far along it is.',
    cameraBookmarks:
        'Jump to a saved view of this scene. The same views are used by the native RTXGI tool, so pictures can be compared side by side.',
    copyLink: 'Copies a link that reopens exactly this view: scene, camera position, GI choice, exposure and split screen.',
    navOrbit: 'Orbit: drag to circle around what you are looking at, scroll to zoom, right-drag to slide sideways.',
    navFly: 'Fly: drag to look around, W A S D to move, Q and E (or C and Space) to go down and up, hold Shift to go faster, scroll to change speed.',
    flySpeed: 'How fast you move in fly mode, in metres per second. Scrolling in fly mode changes this too.',
    fieldOfView: 'How wide the camera sees. Smaller is zoomed in, larger is wide angle. Bookmarks set their own.',
    exposure:
        'Brightens or darkens the whole picture in photographic stops: +1 is twice as bright, -1 half. Each scene opens at its own setting (Sponza +2, because its sun is weak and most of it is in shade). It does not change the lighting itself.',
    exposureReset: "Back to this scene's own exposure.",
    toneMapping:
        'How very bright light is squeezed onto the screen. ACES is the usual film look. AgX and Neutral keep colours truer. Linear clips bright areas to white.',
    referenceCube: 'Shows a 1 metre cube so you can judge the size of the scene.',
    giOff: 'Direct light only: the sun and lamps, plus shadows, but no light bouncing off walls. Shadowed areas go dark.',
    giOn: 'Direct light plus global illumination: light that has bounced around the scene, giving soft fill, colour bleeding and brighter shadows.',
    giOnly:
        'Only the bounced light, with the direct sun and lamps hidden. Use it to see what the GI technique alone contributes. Surfaces still show their colour, and glowing surfaces still glow.',
    technique:
        'Which global illumination technique to use. DDGI is ours: probes in a grid, each tracing rays every frame, so light, sky and moving things update live. The other two come from three.js and are the baselines DDGI has to beat. Light probe grid: light stored at points in the scene, baked once. VXGI: a voxel copy of the scene traced live every frame. All get the same lights, sky and exposure, at full physical strength, so the comparison is fair.',
    ddgiBudget:
        'How much GPU time DDGI may spend per frame updating its probes, in milliseconds, with the GPU working at full speed. DDGI measures once how many probes that buys and then updates that many every frame; changing this works out the new number from what it measured. More updates more probes each frame, so light settles faster after a change; less leaves more time for drawing. 2 ms is the default and the target. The stats box (top left) shows how many probes that buys.',
    ddgiProbes:
        'Draws every DDGI probe as a small ball where it really sits (probes stuck in walls are moved out), coloured by what it is doing. The legend below says what each colour means; the numbers count the probes in each state.',
    ddgiScreenView:
        "Replaces the picture with one of DDGI's check views. GI only: just the bounced light, as it would fall on plain white surfaces. Coverage: green where DDGI answers fully, red where it gives way (outside the probe box, or where no probe can see). Probes near this point: the colours of the 8 probes each point reads from. Turns off while the upscaler is on.",
    ddgiRestart:
        'Throws away what the probes have learned and warms them up again from black, the way a scene starts. Useful to watch how fast the light settles.',
    lpgBounces:
        'How many times light bounces before it is stored in the probes. 0 is direct light only, 1 adds one bounce (the usual choice), 2 adds a second. More bounces take longer to bake.',
    lpgProbes: 'The most probes the grid may use. More probes capture finer lighting changes but bake slower. The spacing is widened to fit.',
    lpgBake: 'Throw the baked lighting away and bake again, with the current settings.',
    vxgiBounces:
        'How many bounces of light are cached in the voxel copy of the scene. 1 is the usual choice. VXGI only bounces the sun and lamps: it takes no light from the sky.',
    wipeToggle:
        'Split the screen between two techniques on the same view. The left side shows one, the right side the other, with a white line between them. Drag the line to compare.',
    wipeLeft:
        'What is drawn left of the white line. "No GI" is direct light only. "Cycles reference" is a path-traced picture of a saved camera view, the right answer to compare against.',
    wipeRight: 'What is drawn right of the white line.',
    compareReference:
        'Split the screen: DDGI on the left, the path-traced Cycles picture of this camera view on the right, at the same exposure and tone mapping. The camera moves to the exact spot, field of view and picture shape the reference was rendered with (dark bars fill the rest of the window), so the two sides line up; move the camera and they no longer do. Its materials are plain matte, so shiny spots differ a little.',
    upscaleToggle:
        'Draws the picture at a lower resolution and rebuilds it to full size with @pmndrs/upscaler, which saves GPU time. Works with No GI and the light probe grid; VXGI has its own anti-aliasing, so this is switched off while VXGI is on screen.',
    upscaleQuality: 'How much smaller the picture is drawn. Quality is a little smaller and looks nearly sharp; Ultra performance is much smaller and softer.',
    statsToggle:
        "Shows the frame time, the GPU time of one frame, and what DDGI is doing (probes updated, its GPU time, memory, the adapter's CPU time). 'What the stats mean', just below, explains each line. GPU times need a GPU that supports timestamps.",
    statsExplain: 'What each line of the stats box (top left of the picture) means.',
    panelHide: 'Hides this panel so the picture fills the window. Press H, or the button at the top right, to bring it back.',
    wipeLine: 'Drag this line left or right to compare the two techniques.',

    // The stats box, top left (each line explains itself on hover).
    statsFrame: 'Time from one drawn frame to the next, and frames per second. 16.7 ms is 60 frames per second.',
    statsGpu: "GPU time of three.js's own passes in one frame (drawing the scene, shadows, the final picture), averaged over the last few frames. DDGI's probe updates are listed separately below.",
    statsProbes:
        'How many DDGI probes were updated in the last frame, out of all of them, and how long it takes at that rate until every probe has had its turn.',
    statsBudget:
        "Why DDGI updates this many probes a frame. When it starts it measures, with the GPU working at full speed, how many probes fit in the update budget, then updates that many every frame; a new budget is worked out from those measurements. It doesn't chase the measured time frame by frame: a GPU with little to do slows down to save power, which makes the same work look slower without costing the frame anything. It updates fewer only when frames arrive late ('slowed'), for example when the GPU gets hot and slows down, and goes back up a while after frames are on time again. It never goes below the number that updates every probe within 4 seconds, except on a slow GPU, where it stops at twice what the budget buys. 'measured cost' is what it measured: a fixed part per frame plus a part per probe. If the scene's shape changes it says so; Start the probes over to measure again.",
    statsDdgiUpdate:
        "GPU time of all of DDGI's passes in a frame: tracing rays from the probes (trace), blending the results into the probes (blend), writing them out for drawing (write), and, while GPU timing is on, a small pass that counts probes per state for these stats (census). The budget you chose is shown beside it. The budget was measured with the GPU at full speed; when the GPU has little to do it slows down, so this can read above the budget while frames are on time. That is expected and costs the frame nothing.",
    statsRefit:
        "GPU time to bring the ray-tracing tree up to date with where every object is, every frame. Cheap for a still scene; it matters once things move.",
    statsMemory:
        "GPU memory the GI library owns: its copy of the scene (positions, normals, materials), the ray-tracing trees, and DDGI's probes. three.js's own memory is not counted.",
    statsSync:
        "CPU time the three.js adapter spends per frame looking for changes in the scene and sending them to the GI library (target: 0.2 ms for a still scene), and then recording the GI work.",
    statsStates: 'How many probes are in each state, counted on the GPU a few frames ago.',
} as const;

export type HelpKey = keyof typeof HELP;

export const DEFAULT_HELP = 'Hover over any control to see what it does.';
