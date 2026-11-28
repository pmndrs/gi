/**
 * `pnpm device:serve`: serves the GPU test pages (including the device page)
 * over HTTPS on the local network, so the Mac, iPad and iOS device can open
 * them and report back.
 *
 * Why HTTPS: browsers only expose WebGPU in a "secure context". `localhost`
 * counts as one, but `http://192.168.x.x` does not, so a phone opening the
 * laptop by its network address would see no `navigator.gpu`. A self-signed
 * certificate is enough; the visitor taps through a one-time warning.
 *
 * The certificate is made with `openssl` (Git for Windows ships one) and kept
 * in `node_modules/.cache/ddgi-lan/`, which git already ignores. It is made
 * again when the machine's network addresses change or it gets old.
 *
 * The device page POSTs its results to `/__device-results`; this server
 * appends them to `bench/results/incoming/timing-<date>.jsonl`.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync, appendFileSync, statSync } from 'node:fs';
import { networkInterfaces, type NetworkInterfaceInfo } from 'node:os';
import { join } from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { createServer, type ViteDevServer } from 'vite';

const GPU_TOOLS_DIR = import.meta.dirname;
const REPO_ROOT = join(GPU_TOOLS_DIR, '..', '..');
const CERTIFICATE_DIR = join(REPO_ROOT, 'node_modules', '.cache', 'ddgi-lan');
const INCOMING_RESULTS_DIR = join(REPO_ROOT, 'bench', 'results', 'incoming');

const DEFAULT_PORT = 5291;
const DEVICE_PAGE_PATH = '/pages/device/index.html';
const RESULTS_ENDPOINT = '/__device-results';
const MAX_RESULT_BODY_BYTES = 1_000_000;
/**
 * The server is reachable by anyone on the network, so what it can be made to
 * write is limited: when the incoming folder holds this much, new results are refused.
 */
const MAX_INCOMING_FOLDER_BYTES = 20_000_000;
/** The only folders the page may read through vite: the test pages and the code they import. */
const SERVED_FOLDERS = [GPU_TOOLS_DIR, join(REPO_ROOT, 'packages', 'core', 'src')];
/** Apple rejects certificates valid for more than 398 days; a month is plenty for a test session. */
const CERTIFICATE_DAYS = 30;
const CERTIFICATE_MAX_AGE_MS = (CERTIFICATE_DAYS - 2) * 24 * 60 * 60 * 1000;

/** IPv4 addresses other devices on the network can reach this machine at. */
export function lanAddresses(
    interfaces: NodeJS.Dict<NetworkInterfaceInfo[]> = networkInterfaces(),
): string[] {
    const addresses: string[] = [];
    for (const infos of Object.values(interfaces)) {
        for (const info of infos ?? []) {
            if (info.family === 'IPv4' && !info.internal) addresses.push(info.address);
        }
    }
    return addresses;
}

/** The URLs to type on each device, one per address of this machine. */
export function devicePageUrls(addresses: readonly string[], port: number): string[] {
    return addresses.map((address) => `https://${address}:${port}${DEVICE_PAGE_PATH}`);
}

/**
 * An openssl config file for a self-signed server certificate that is valid
 * for localhost and every given address. Browsers ignore the common name and
 * only check the subject alternative names, so the addresses must be listed.
 * Writing our own file also means we don't depend on where this openssl build
 * looks for its default config, which is unreliable on Windows.
 */
export function buildOpenSslConfig(addresses: readonly string[]): string {
    const alternativeNames = [
        'DNS.1 = localhost',
        'IP.1 = 127.0.0.1',
        ...addresses.map((address, index) => `IP.${index + 2} = ${address}`),
    ];
    return [
        '[req]',
        'distinguished_name = subject',
        'x509_extensions = server_extensions',
        'prompt = no',
        '',
        '[subject]',
        'CN = ddgi-lan',
        '',
        '[server_extensions]',
        'basicConstraints = critical, CA:FALSE',
        'keyUsage = critical, digitalSignature, keyEncipherment',
        'extendedKeyUsage = serverAuth',
        'subjectAltName = @alternative_names',
        '',
        '[alternative_names]',
        ...alternativeNames,
        '',
    ].join('\n');
}

function findOpenSsl(): string {
    const programFiles = process.env.ProgramFiles ?? 'C:/Program Files';
    const candidates = [
        'openssl',
        join(programFiles, 'Git', 'usr', 'bin', 'openssl.exe'),
        join(programFiles, 'Git', 'mingw64', 'bin', 'openssl.exe'),
    ];
    for (const candidate of candidates) {
        try {
            execFileSync(candidate, ['version'], { stdio: 'ignore' });
            return candidate;
        } catch {
            // Not there (or not runnable); try the next one.
        }
    }
    throw new Error(
        'openssl was not found. Install Git for Windows (it includes openssl) or put openssl on the PATH. ' +
            `Looked for: ${candidates.join(', ')}`,
    );
}

interface Certificate {
    key: string;
    cert: string;
    certPath: string;
}

function ensureCertificate(addresses: readonly string[]): Certificate {
    mkdirSync(CERTIFICATE_DIR, { recursive: true });
    const keyPath = join(CERTIFICATE_DIR, 'key.pem');
    const certPath = join(CERTIFICATE_DIR, 'cert.pem');
    const configPath = join(CERTIFICATE_DIR, 'openssl.cnf');

    const config = buildOpenSslConfig(addresses);
    const upToDate =
        existsSync(keyPath) &&
        existsSync(certPath) &&
        existsSync(configPath) &&
        readFileSync(configPath, 'utf8') === config &&
        Date.now() - statSync(certPath).mtimeMs < CERTIFICATE_MAX_AGE_MS;

    if (!upToDate) {
        writeFileSync(configPath, config);
        execFileSync(
            findOpenSsl(),
            [
                'req', '-x509',
                '-newkey', 'rsa:2048',
                '-nodes',
                '-keyout', keyPath,
                '-out', certPath,
                '-days', String(CERTIFICATE_DAYS),
                '-config', configPath,
            ],
            { stdio: 'ignore' },
        );
        console.log(`Made a new self-signed certificate in ${CERTIFICATE_DIR}`);
    }
    return { key: readFileSync(keyPath, 'utf8'), cert: readFileSync(certPath, 'utf8'), certPath };
}

/** Total size of the files directly inside a folder; 0 if it does not exist yet. */
export function folderBytes(folder: string): number {
    if (!existsSync(folder)) return 0;
    return readdirSync(folder).reduce((total, name) => {
        const stats = statSync(join(folder, name));
        return total + (stats.isFile() ? stats.size : 0);
    }, 0);
}

/** Whether adding `incomingBytes` to a folder that already holds `currentBytes` would pass the cap. */
export function exceedsStorageCap(currentBytes: number, incomingBytes: number, capBytes = MAX_INCOMING_FOLDER_BYTES): boolean {
    return currentBytes + incomingBytes > capBytes;
}

/** A one-line description of a device page result, for the server's terminal. */
export function summarizeResult(payload: unknown): string {
    const result = payload as {
        label?: string;
        status?: string;
        environment?: { userAgent?: string; adapterAvailable?: boolean };
        run?: { timestampsOnDevice?: boolean; overheadMs?: number | null } | null;
    };
    const timestamps = result.run?.timestampsOnDevice;
    const overhead = result.run?.overheadMs;
    return [
        `label=${result.label || '(none)'}`,
        `status=${result.status ?? '?'}`,
        `WebGPU=${result.environment?.adapterAvailable ? 'yes' : 'no'}`,
        `timestamps=${timestamps === undefined ? '?' : timestamps ? 'yes' : 'no'}`,
        `overhead=${typeof overhead === 'number' ? `${overhead} ms` : 'n/a'}`,
    ].join(' ');
}

function readBody(request: IncomingMessage): Promise<string> {
    return new Promise((resolvePromise, rejectPromise) => {
        const chunks: Buffer[] = [];
        let size = 0;
        request.on('data', (chunk: Buffer) => {
            size += chunk.length;
            if (size > MAX_RESULT_BODY_BYTES) {
                rejectPromise(new Error('Result body is too large.'));
                request.destroy();
                return;
            }
            chunks.push(chunk);
        });
        request.on('end', () => resolvePromise(Buffer.concat(chunks).toString('utf8')));
        request.on('error', rejectPromise);
    });
}

async function handleResults(request: IncomingMessage, response: ServerResponse): Promise<void> {
    if (request.method !== 'POST') {
        response.statusCode = 405;
        response.end('POST only');
        return;
    }
    try {
        const body = await readBody(request);
        const payload: unknown = JSON.parse(body);
        if (typeof payload !== 'object' || payload === null) throw new Error('Expected a JSON object.');
        if (exceedsStorageCap(folderBytes(INCOMING_RESULTS_DIR), body.length)) {
            response.statusCode = 507;
            response.end('The incoming results folder is full. Delete old files in bench/results/incoming/.');
            return;
        }
        mkdirSync(INCOMING_RESULTS_DIR, { recursive: true });
        const date = new Date().toISOString().slice(0, 10);
        const record = {
            receivedAt: new Date().toISOString(),
            remoteAddress: request.socket.remoteAddress ?? null,
            payload,
        };
        appendFileSync(join(INCOMING_RESULTS_DIR, `timing-${date}.jsonl`), `${JSON.stringify(record)}\n`);
        console.log(`Result from ${record.remoteAddress}: ${summarizeResult(payload)}`);
        response.statusCode = 204;
        response.end();
    } catch (error) {
        response.statusCode = 400;
        response.end(error instanceof Error ? error.message : 'Bad request');
    }
}

async function main(): Promise<void> {
    const addresses = lanAddresses();
    if (addresses.length === 0) {
        console.warn('No network address found: only this machine can open the page. Connect to Wi-Fi or Ethernet.');
    }
    const certificate = ensureCertificate(addresses);

    const server: ViteDevServer = await createServer({
        root: GPU_TOOLS_DIR,
        // Self-contained, like the test runner's server: no repo vite config.
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: {
            host: '0.0.0.0',
            port: DEFAULT_PORT,
            strictPort: false,
            https: { key: certificate.key, cert: certificate.cert },
            // Anything outside these folders is refused, even though the server listens on the whole network.
            fs: { strict: true, allow: SERVED_FOLDERS },
            // Reached by IP address or a .local name; the default allow-list would refuse names.
            allowedHosts: true,
            // Hot reload would open a secure websocket that a phone with an untrusted certificate refuses.
            hmr: false,
            watch: null,
        },
        plugins: [
            {
                name: 'device-page-routes',
                configureServer(devServer) {
                    devServer.middlewares.use('/favicon.ico', (_request, response) => {
                        response.statusCode = 204;
                        response.end();
                    });
                    devServer.middlewares.use(RESULTS_ENDPOINT, (request, response) => {
                        void handleResults(request, response);
                    });
                    // Typing just the address should land on the device page.
                    devServer.middlewares.use((request, response, next) => {
                        if (request.url !== '/') {
                            next();
                            return;
                        }
                        response.statusCode = 302;
                        response.setHeader('location', DEVICE_PAGE_PATH);
                        response.end();
                    });
                },
            },
        ],
    });
    await server.listen();

    const address = server.httpServer?.address();
    const port = typeof address === 'object' && address ? address.port : DEFAULT_PORT;
    const urls = devicePageUrls(addresses, port);
    const examples = urls.length > 0 ? urls : [`https://localhost:${port}${DEVICE_PAGE_PATH}`];

    console.log('');
    console.log('Device page server is running (HTTPS, self-signed certificate).');
    console.log('');
    console.log('Open this on each device, on the same Wi-Fi as this PC:');
    for (const url of examples) console.log(`  ${url}?label=ipad-safari`);
    console.log('');
    console.log('Change label= to name the device (mac-chrome, mac-safari, ipad-safari, ios-safari).');
    console.log('The browser warns that the connection is not private. That is expected:');
    console.log('  Safari: Show Details, then "visit this website", then confirm.');
    console.log('  Chrome: Advanced, then "Proceed to ... (unsafe)".');
    console.log('The page runs by itself and shows a table. Results are also sent here and saved to');
    console.log(`  ${join(INCOMING_RESULTS_DIR, 'timing-<date>.jsonl')}`);
    console.log('If a device cannot connect, allow Node.js through the Windows firewall for private networks.');
    console.log(`Certificate (for installing on a device if the warning cannot be skipped): ${certificate.certPath}`);
    console.log('Press Ctrl-C to stop.');

    const stop = (): void => {
        void server.close().finally(() => process.exit(0));
    };
    process.on('SIGINT', stop);
    process.on('SIGTERM', stop);
}

if (import.meta.main) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
