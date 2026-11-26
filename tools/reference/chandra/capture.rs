//! Headless consumer of Chandra at 9af00e8. Copy to the upstream src/bin/ directory.
//! Captures the default single-bounce Split RC path on the shared Sponza camera.
use chandra::algorithm::rc::{over, RcData};
use chandra::algorithm::splat::SplatData;
use chandra::algorithm::trace::TraceData;
use chandra::algorithm::{CommandList, CoreData};
use chandra::scene::Scene;
use keter::lang::types::vector::Vec3;
use keter::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use yesod::camera::Camera;
use yesod::utils::dispatch0;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: u32 = 128;

fn write_rgb(directory: &Path, name: &str, pixels: Vec<Vec3<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = BufWriter::new(File::create(directory.join(format!("{name}.rgb32f")))?);
    let mut preview = image::RgbImage::new(WIDTH, HEIGHT);
    for (pixel, p) in preview.pixels_mut().zip(pixels) {
        for (channel, value) in [p.x, p.y, p.z].into_iter().enumerate() {
            if !value.is_finite() { return Err(format!("{name} contains a non-finite radiance").into()); }
            file.write_all(&value.to_le_bytes())?;
            let positive = value.max(0.);
            let mapped = positive / (1. + positive);
            let srgb = if mapped <= 0.0031308 { 12.92 * mapped } else { 1.055 * mapped.powf(1. / 2.4) - 0.055 };
            pixel[channel] = (srgb * 255.).round() as u8;
        }
    }
    file.flush()?;
    preview.save(directory.join(format!("{name}.png")))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args().nth(1).expect("capture <output directory>");
    let directory = Path::new(&directory);
    fs::create_dir_all(directory)?;
    let mut scene = Scene::load_file("scenes/sponza.json");
    let black = directory.join("black.png");
    image::RgbImage::new(1, 1).save(&black)?;
    scene.world.envmap = black;
    scene.world.envmap_scale = 0.;
    scene.world.sun_color = glam::Vec3::ZERO;
    yesod::printer::global::set_capacity(1024 * 1024);
    let core = CoreData::new([WIDTH, HEIGHT], scene.world);
    let mut rc = RcData::new(scene.rc_config);
    let trace = TraceData::new(&core, &rc, (WIDTH * HEIGHT * 3) as usize);
    let splat = SplatData::new(&core, &rc, 1.);
    // The interactive main constructs DDGI after SplatData; that initialization commits
    // SplatData's newly registered irradiance texture to the shared bindless table.
    keter::bindless::BINDLESS.update();
    let camera = Camera {
        screen_size: glam::Vec2::new(WIDTH as f32, HEIGHT as f32),
        yaw: scene.camera.yaw, pitch: scene.camera.pitch, pos: scene.camera.pos, fov: 1.15,
    };
    let c0_len = ((2 * rc.base_angular_resolution.pow(2)) as f32 / (4. * std::f32::consts::PI)).sqrt();
    assert_eq!((rc.base_angular_resolution, rc.cm1_len, rc.num_cascades), (4, 0., 4));
    for tick in 0..FRAMES {
        let mut commands = CommandList::new();
        if tick == 0 { rc.reset_table(&mut commands, false); }
        commands += core.view.write_host(camera.view());
        commands += core.compute_gbuffer();
        commands += rc.cache.set_view(camera.view());
        commands += rc.init_probes_base(&core);
        for level in 1..rc.num_cascades { commands += rc.init_probes(level); }
        commands += rc.set_lifetime_base(&core);
        for level in 1..rc.num_cascades { commands += rc.set_lifetime(level); }
        rc.reset_table(&mut commands, false);
        commands += rc.compute_average_position(&core);
        commands += trace.reset();
        commands += trace.compute_ray_count_base(&core, &rc);
        for level in 1..rc.num_cascades { commands += trace.compute_ray_count(&rc, level); }
        for level in (0..rc.num_cascades).rev() { commands += trace.compute_ray_offset(&rc, level); }
        commands += trace.compute_rays(&core, &rc, tick, c0_len, false);
        commands.execute();
        let ray_count = trace.next_traced_ray.read_blocking() as usize;
        assert!(ray_count <= trace.traced_rays.len());
        commands += rc.deposit_split(trace.traced_rays.view(..ray_count));
        for level in (0..rc.num_cascades).rev() { commands += rc.merge_cache(level); }
        commands += splat.reset();
        commands += splat.calculate_irradiance(&rc);
        commands += splat.copy_irradiance_borders(&rc);
        commands += splat.shade_optimized_32(&core, &rc);
        commands.execute();
        if tick % 32 == 0 || tick + 1 == FRAMES { println!("Frame {}: {} rays", tick + 1, ray_count); }
    }

    let n = (WIDTH * HEIGHT) as usize;
    let diffuse = DEVICE.create_buffer::<Vec3<f32>>(n);
    let specular = DEVICE.create_buffer::<Vec3<f32>>(n);
    let labels = DEVICE.create_buffer::<u32>(n);
    let mut output = CommandList::new();
    let view = camera.view();
    output.push(dispatch0(track!(|| {
        let xy = dispatch_id().xy();
        let i = core.pixel_to_linear(xy);
        let depth = core.read_depth(xy);
        let zero = Vec3::splat_expr(0.0_f32);
        diffuse.write(i, zero);
        specular.write(i, zero);
        labels.write(i, 0_u32);
        if depth > 0.0 {
            let hit = core.hit.read(i);
            diffuse.write(i, splat.get_radiance(i) * core.world.get_diffuse(hit));
            labels.write(i, (hit.inst == 1u32).select(2u32.expr(), (hit.inst == 9u32).select(3u32.expr(), 1u32.expr())));
            let direction = view.expr().ray_dir(xy.cast_f32() + 0.5).normalize();
            let point = view.pos.expr() + direction * depth;
            let reflected = chandra::utils::reflect(direction, core.read_normal(xy));
            let incoming = rc.sample(point, 3u32.expr(), reflected).var();
            for level in [2u32, 1, 0] {
                *incoming = over(rc.cache.sample(point, level.expr(), reflected), **incoming);
            }
            if incoming.is_finite().all() { specular.write(i, **incoming); }
        }
    }), [WIDTH, HEIGHT, 1]));
    output.execute();
    write_rgb(directory, "diffuse", diffuse.copy_to_vec())?;
    write_rgb(directory, "specular", specular.copy_to_vec())?;
    fs::write(directory.join("primary.u8"), labels.copy_to_vec().into_iter().map(|n| n as u8).collect::<Vec<_>>())?;
    Ok(())
}
