// Cached camera-path layer. Original map assets and Entity stay resident in this Viewer.
export function createT07PathLayer({ app, Asset, Entity, baseEntity, camera, url }) {
    const resolved = new URL(url, location.href);
    if (resolved.origin !== location.origin) throw new Error('The path layer must use a local same-origin asset.');
    let entity = null;
    let asset = null;
    let loading = null;
    let disposed = false;
    let rejectLoad = null;
    const pendingFrames = new Set();
    const stats = { loads: 0, toggles: 0, lastToggleMs: 0 };

    const status = () => ({
        visible: !!entity?.enabled, cached: !!entity, loads: stats.loads,
        toggles: stats.toggles, lastToggleMs: stats.lastToggleMs,
        baseAssetId: String(baseEntity.gsplat.asset),
        baseEntityId: baseEntity.getGuid(),
        baseResourceResident: !!baseEntity.gsplat.resource,
        pathAssetId: asset ? String(asset.id) : null
    });

    const ensureLoaded = () => {
        if (disposed) return Promise.reject(new Error('This path-layer Viewer has been closed.'));
        if (entity) return Promise.resolve();
        if (loading) return loading;
        stats.loads++;
        loading = new Promise((resolve, reject) => {
            rejectLoad = reject;
            asset = new Asset('T07 camera path', 'gsplat', {
                url: resolved.href, filename: resolved.pathname.split('/').pop()
            });
            asset.once('load', () => {
                if (disposed) return;
                entity = new Entity('T07 camera path layer');
                entity.enabled = false;
                // Match the existing map transform exactly, including its Z-180 rotation.
                entity.setLocalPosition(baseEntity.getLocalPosition());
                entity.setLocalRotation(baseEntity.getLocalRotation());
                entity.setLocalScale(baseEntity.getLocalScale());
                entity.addComponent('gsplat', { unified: true, asset });
                app.root.addChild(entity);
                rejectLoad = null;
                resolve();
            });
            asset.once('error', error => {
                const failed = asset;
                asset = null;
                loading = null;
                rejectLoad = null;
                if (failed) { failed.unload(); app.assets.remove(failed); }
                reject(new Error(`Could not load the camera path layer: ${String(error)}`));
            });
            app.assets.add(asset);
            app.assets.load(asset);
        });
        return loading;
    };

    // Wait for the unified renderer's revised placement set and sorting to settle.
    // Saving PNG is disabled by the host while this promise is pending.
    const renderVisibility = visible => new Promise((resolve, reject) => {
        const events = app.systems.gsplat;
        let timer;
        let finished = false;
        const cleanup = () => {
            events.off('frame:ready', onReady);
            app.off('frameend', onFrameEnd);
            clearTimeout(timer);
            pendingFrames.delete(cancel);
        };
        const done = error => {
            if (finished) return;
            finished = true;
            cleanup();
            if (error) reject(error); else resolve();
        };
        const cancel = () => done(new Error('The path-layer Viewer was closed during an update.'));
        const onFrameEnd = () => done();
        const onReady = (readyCamera, _layer, ready, count) => {
            if (readyCamera !== camera.camera || !ready || count !== 0) return;
            events.off('frame:ready', onReady);
            app.once('frameend', onFrameEnd);
        };
        pendingFrames.add(cancel);
        events.on('frame:ready', onReady);
        timer = setTimeout(() => done(new Error('Camera path rendering timed out.')), 60000);
        entity.enabled = visible;
        app.renderNextFrame = true;
    });

    const setVisible = async visible => {
        const started = performance.now();
        if (disposed) throw new Error('This path-layer Viewer has been closed.');
        if (visible) await ensureLoaded();
        if (entity && entity.enabled !== !!visible) {
            const previous = entity.enabled;
            try { await renderVisibility(!!visible); }
            catch (error) {
                if (!disposed) { entity.enabled = previous; app.renderNextFrame = true; }
                throw error;
            }
            stats.toggles++;
        }
        stats.lastToggleMs = performance.now() - started;
        return status();
    };

    const dispose = () => {
        if (disposed) return;
        disposed = true;
        for (const cancel of [...pendingFrames]) cancel();
        rejectLoad?.(new Error('The path-layer Viewer was closed during loading.'));
        rejectLoad = null;
        entity?.destroy();
        entity = null;
        if (asset) { asset.unload(); app.assets.remove(asset); asset = null; }
        loading = null;
        app.renderNextFrame = true;
        window.removeEventListener('pagehide', dispose);
    };
    window.addEventListener('pagehide', dispose, { once: true });
    return { setVisible, status, dispose };
}
