import type { NetworkInterfaceInfo } from 'node:os';
import { describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildOpenSslConfig, devicePageUrls, exceedsStorageCap, folderBytes, lanAddresses, summarizeResult } from './serve-lan.ts';

function info(address: string, family: 'IPv4' | 'IPv6', internal: boolean): NetworkInterfaceInfo {
    return { address, family, internal } as NetworkInterfaceInfo;
}

describe('lanAddresses', () => {
    it('keeps external IPv4 addresses only', () => {
        const addresses = lanAddresses({
            Loopback: [info('127.0.0.1', 'IPv4', true)],
            'Wi-Fi': [info('fe80::1', 'IPv6', false), info('192.168.1.20', 'IPv4', false)],
            Ethernet: undefined,
        });
        expect(addresses).toEqual(['192.168.1.20']);
    });
});

describe('devicePageUrls', () => {
    it('builds one https URL per address', () => {
        expect(devicePageUrls(['10.0.0.5', '192.168.1.20'], 5291)).toEqual([
            'https://10.0.0.5:5291/pages/device/index.html',
            'https://192.168.1.20:5291/pages/device/index.html',
        ]);
    });
});

describe('buildOpenSslConfig', () => {
    it('lists localhost, loopback and every address as alternative names', () => {
        const config = buildOpenSslConfig(['10.0.0.5', '192.168.1.20']);
        expect(config).toContain('DNS.1 = localhost');
        expect(config).toContain('IP.1 = 127.0.0.1');
        expect(config).toContain('IP.2 = 10.0.0.5');
        expect(config).toContain('IP.3 = 192.168.1.20');
        expect(config).toContain('serverAuth');
    });
});

describe('summarizeResult', () => {
    it('describes a full result in one line', () => {
        const line = summarizeResult({
            label: 'ipad',
            status: 'pass',
            environment: { adapterAvailable: true },
            run: { timestampsOnDevice: false, overheadMs: null },
        });
        expect(line).toBe('label=ipad status=pass WebGPU=yes timestamps=no overhead=n/a');
    });
    it('copes with a run that never measured', () => {
        expect(summarizeResult({ status: 'fail', environment: { adapterAvailable: false }, run: null })).toContain(
            'WebGPU=no',
        );
    });
});

describe('storage cap', () => {
    it('refuses a write that would pass the cap and allows one that reaches it exactly', () => {
        expect(exceedsStorageCap(900, 100, 1000)).toBe(false);
        expect(exceedsStorageCap(900, 101, 1000)).toBe(true);
    });

    it('measures the files in a folder, and a missing folder as empty', () => {
        const folder = mkdtempSync(join(tmpdir(), 'ddgi-cap-'));
        try {
            writeFileSync(join(folder, 'a.jsonl'), 'x'.repeat(10));
            writeFileSync(join(folder, 'b.jsonl'), 'y'.repeat(5));
            expect(folderBytes(folder)).toBe(15);
            expect(folderBytes(join(folder, 'missing'))).toBe(0);
        } finally {
            rmSync(folder, { recursive: true, force: true });
        }
    });
});
