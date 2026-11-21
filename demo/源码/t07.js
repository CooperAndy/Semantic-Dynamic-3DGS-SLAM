(() => {
  'use strict';
  let locked = null;
  let busy = false;
  let lastEvaluation = null;
  let pendingCamera = null;
  const $ = id => document.getElementById(id);
  const status = text => { $('t07-status').textContent = text; };
  window.t07ReferenceAvailability = () => {
    $('t07-reference-frame').disabled = busy || mode !== 'rgb' || isCameraPathVisible() || !viewerReady;
  };
  // Already-running local servers may still have the previous evaluator loaded.
  const requestError = message => ({
    '无效场景或帧号': 'Invalid scene or frame number.',
    '帧号超出当前场景范围': 'Frame number is outside the selected scene range.',
    '缺少该帧原始 RGB 图像，无法评分': 'The original RGB image for this frame is missing; evaluation is unavailable.',
    '渲染图和参考图 RGB 尺寸不一致；禁止缩放后评分': 'Rendered and reference RGB dimensions differ. Resizing before evaluation is not allowed.',
    'PSNR 仅评测完整 RGB 地图，请切回 RGB 地图': 'PSNR evaluates the full RGB map only. Switch to RGB Map.',
    '必须先跳转并锁定当前参考帧，不能给自由视角评分': 'Align and lock a reference-frame camera before evaluation. Free views cannot be scored.',
    '渲染相机矩阵与该参考帧不一致': 'The rendered camera matrix does not match the reference frame.',
    '评分 PNG 必须与原始参考图尺寸一致': 'The evaluated PNG must have the original reference-image dimensions.'
  })[message] || message;
  const bridge = () => {
    if (!viewerReady || !frame.contentWindow?.t07Camera || !frame.contentWindow?.captureFrame)
      throw new Error('The map is still loading. Please try again shortly.');
    return frame.contentWindow;
  };
  function validate(doc) {
    const c = doc?.camera;
    if (doc?.schema !== 't07-camera-v1' || doc.sceneId !== config.sceneId)
      throw new Error('Camera file format or scene mismatch. Select the matching scene first.');
    if (doc.sourceFrames !== undefined && (!Array.isArray(doc.sourceFrames) || doc.sourceFrames.length !== 2 || !doc.sourceFrames.every((value, index) => value === config.sourceFrames[index])))
      throw new Error('The saved camera frame range does not match the current scene.');
    if (!c || !Array.isArray(c.position) || c.position.length !== 3 ||
        !Array.isArray(c.angles) || c.angles.length !== 3 ||
        ![...c.position, ...c.angles, c.distance, c.fov, doc.aspect].every(Number.isFinite) ||
        c.distance < 0 || c.fov <= 0 || c.fov >= 179 || doc.aspect < 0.1 || doc.aspect > 10 ||
        !['orbit', 'fly', 'walk'].includes(c.mode)) throw new Error('Invalid camera parameters.');
    if (c.reference) {
      const r = c.reference;
      if (r.sceneId !== config.sceneId || !Number.isInteger(r.frame) || !poses.has(r.frame) ||
          ![r.width,r.height,r.fx,r.fy,r.cx,r.cy].every(Number.isFinite) ||
          r.width <= 0 || r.height <= 0 || r.fx <= 0 || r.fy <= 0 ||
          Math.abs(doc.aspect-r.width/r.height)>1e-6) throw new Error('Invalid reference-camera parameters.');
    }
    return doc;
  }
  function snapshot() {
    const win = bridge();
    const camera = win.t07Camera.get();
    // Snapshot current rendered camera, then freeze both input updates and smoothing.
    const doc = validate({schema: 't07-camera-v1', sceneId: config.sceneId,
      coordinates: 'Viewer world coordinates; exported KITTI transform [-x,-y,z]',
      aspect: camera.reference ? camera.reference.width / camera.reference.height : camera.canvasWidth / camera.canvasHeight, camera});
    locked = structuredClone(doc);
    win.t07Camera.lock(locked.camera);
    frame.style.pointerEvents = 'none';
    status('View locked. Save a PNG from this camera pose.');
    return locked;
  }
  window.t07PrepareMapReload = () => {
    pendingCamera = structuredClone(bridge().t07Camera.get());
  };
  window.t07RestoreLocked = () => {
    if (locked?.sceneId === config.sceneId) {
      bridge().t07Camera.lock(locked.camera);
      frame.style.pointerEvents = 'none';
      if (locked.camera.reference) {
        fitReference();
        if (mode === 'rgb' && !isCameraPathVisible() && !vehicleCleanupEnabled && !vehicleCleanupHighlight) setTimeout(() => referenceAction(), 0);
      }
    } else if (pendingCamera) {
      bridge().t07Camera.lock(pendingCamera);
      bridge().t07Camera.unlock();
      frame.style.pointerEvents = '';
    } else applyDefaultCamera();
    pendingCamera = null;
  };
  function metadata() {
    if (!locked) snapshot();
    const diagnostic = !isCameraPathVisible() && !vehicleCleanupEnabled && !vehicleCleanupHighlight && lastEvaluation && lastEvaluation.sceneId === config.sceneId && lastEvaluation.variant === variant && mode === 'rgb' && lastEvaluation.frame === locked.camera.reference?.frame && lastEvaluation.supersample === Number($('t07-ss').value) ? lastEvaluation : null;
    return {...structuredClone(locked), ...(diagnostic ? {referenceDiagnostic:diagnostic} : {}), capturedAt: new Date().toISOString(),
      sceneLabel: selectedScene.label,
      sourceFrames: [...config.sourceFrames],
      variant: mode === 'semantic' ? config.semanticMapId : variant,
      rgbVariantForReturn: mode === 'semantic' ? variant : undefined,
      mapRelation: mode === 'semantic' ? (vehicleCleanupEnabled
        ? 'Vehicle-cleaned subset of F1 semantic Gaussians; retained rows and colors unchanged'
        : config.semanticRelation) : undefined,
      motionReference: mode === 'status' && motionSnapshot?.scene_id === config.sceneId
        ? {frame: motionSnapshot.reference_frame, model: motionSnapshot.motion_model,
           labelMode: motionSnapshot.label_mode || 'per_frame',
           sourceFrames: motionSnapshot.source_frames,
           aggregationConfig: motionSnapshot.aggregation_config,
           sourceMapSha256: motionSnapshot.source_map_sha256,
           note: 'Estimated label positions are HTML overlays; exported PNG contains the Gaussian map. Object-level map binding and occlusion are not verified.',
           objects: motionSnapshot.objects} : undefined,
      dynamicVehicleRemoval: {enabled:vehicleCleanupEnabled && !vehicleCleanupHighlight,
        requestedEnabled:vehicleCleanupEnabled,highlightPreview:vehicleCleanupHighlight,
        removedCount:vehicleCleanupEnabled && !vehicleCleanupHighlight ? config.dynamicVehicleRemoval.removedCount : 0,
        highlightedCount:vehicleCleanupHighlight ? config.dynamicVehicleRemoval.removedCount : 0,
        maskSha256:vehicleCleanupEnabled || vehicleCleanupHighlight ? config.dynamicVehicleRemoval.maskSha256 : null,
        scope:'Predicted vehicle-only display cleanup; original map preserved'},
      mode, cameraPathVisible: isCameraPathVisible(),
      cameraPathLayer: usesCachedPathLayer() && isCameraPathVisible() ? {
        asset: resolveAsset(config.cameraPathLayer.asset), manifest: resolveAsset(config.cameraPathLayer.manifest),
        sha256: config.cameraPathLayer.sha256, gaussianCount: config.cameraPathLayer.gaussianCount,
        method: 'Cached independent Gaussian layer; original RGB asset retained'
      } : undefined,
      assetUrl: resolveAsset(activeMapAsset()), assetManifest: resolveAsset(activeMapManifest()),
      poseProtocol: config.poseProtocol,
      note: 'Browser Gaussian renderer export; no UI. Not a SplaTAM metric render or quality improvement claim.'};
  }
  async function post(path, value) {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
    const result = await response.json();
    if (!response.ok) throw new Error(requestError(result.error) || 'Save failed');
    return result;
  }
  function action(fn) {
    return async () => {
      if (busy) return;
      busy = true;
      $('t07-reference-frame').disabled = true;
      const buttons = [...document.querySelectorAll('[id^="t07-"]')].filter(e => e.tagName === 'BUTTON');
      const switches = [sceneSelect, $('camera-path'), $('motion-frame'), $('vehicle-cleanup'), $('vehicle-cleanup-highlight'), ...document.querySelectorAll('.tabs button')];
      const prior = switches.map(e => e.disabled);
      buttons.forEach(e => { e.disabled = true; });
      switches.forEach(e => { e.disabled = true; });
      try { await fn(); } catch (error) { status(`Operation failed: ${error.message}`); }
      finally {
        buttons.forEach(e => { e.disabled = false; });
        $('t07-restore-apply').disabled = !$('t07-restore-select').value;
        switches.forEach((e, i) => { e.disabled = prior[i]; });
        busy = false;
        window.t07ReferenceAvailability();
      }
    };
  }
  window.t07RunExclusive = action;
  $('t07-lock').onclick = action(snapshot);
  $('t07-unlock').onclick = action(() => {
    bridge().t07Camera.unlock(); locked = null; frame.style.pointerEvents = '';
    fitReference(); window.t07ReferenceInvalidate();
    status('View unlocked. Navigate freely; saving will lock the current view.');
  });
  let savedCaptures = [];
  function restoreCamera(candidate, label) {
    bridge();
    locked = structuredClone(validate(candidate));
    window.t07RestoreLocked();
    window.t07ReferenceInvalidate();
    status(`Capture view restored and locked (${label}). Click Unlock View to navigate.`);
  }
  function updateCapturePreview() {
    const capture = savedCaptures.find(item => item.name === $('t07-restore-select').value);
    const preview = $('t07-restore-preview');
    preview.hidden = !capture?.previewUrl;
    if (capture?.previewUrl) preview.src = capture.previewUrl;
    else preview.removeAttribute('src');
    $('t07-restore-apply').disabled = !capture;
  }
  $('t07-import').onclick = action(async () => {
    bridge();
    const query = new URLSearchParams({scene:config.sceneId, start:String(config.sourceFrames[0]), end:String(config.sourceFrames[1])});
    const response = await fetch(`/api/captures?${query}`);
    if (!response.ok) throw new Error('Could not load saved captures. Close and reopen the Demo to update its local service.');
    const result = await response.json();
    savedCaptures = result.captures;
    $('t07-restore-root').textContent = `Current export folder: ${result.root}`;
    $('t07-restore-select').replaceChildren(...savedCaptures.map(capture => {
      const option = document.createElement('option');
      option.value = capture.name;
      option.textContent = capture.name;
      return option;
    }));
    $('t07-restore-select').disabled = !savedCaptures.length;
    $('t07-restore-empty').hidden = !!savedCaptures.length;
    updateCapturePreview();
    $('t07-restore-dialog').showModal();
  });
  $('t07-restore-select').onchange = updateCapturePreview;
  $('t07-restore-cancel').onclick = () => $('t07-restore-dialog').close();
  $('t07-restore-browse').onclick = action(async () => {
    bridge();
    $('t07-restore-dialog').close();
    status('Choose a camera JSON file in the local selection window. It starts in the current Demo export folder.');
    const result = await post('/api/pick-camera', {});
    if (result.cancelled) {
      status('Camera selection cancelled.');
      $('t07-restore-dialog').showModal();
      return;
    }
    restoreCamera(result.metadata, result.name);
  });
  $('t07-restore-apply').onclick = action(async () => {
    const capture = savedCaptures.find(item => item.name === $('t07-restore-select').value);
    if (!capture) throw new Error('Choose a saved capture first.');
    const response = await fetch(capture.cameraUrl);
    if (!response.ok) throw new Error('The saved camera file is no longer available. Reopen Restore Camera to refresh the list.');
    restoreCamera(await response.json(), capture.name);
    $('t07-restore-dialog').close();
  });
  function fitReference() {
    const ref = locked?.camera.reference;
    if (!ref) {
      frame.style.width = ''; frame.style.height = ''; frame.style.top = ''; frame.style.left = ''; frame.style.transform = '';
      return;
    }
    const box = document.getElementById('viewport').getBoundingClientRect();
    const width = Math.min(box.width, box.height*ref.width/ref.height);
    frame.style.width = `${width}px`; frame.style.height = `${width*ref.height/ref.width}px`;
    frame.style.top = '50%'; frame.style.left = '50%'; frame.style.transform = 'translate(-50%, -50%)';
  }
  window.t07ResetForMotionFrame = () => {
    if (viewerReady && frame.contentWindow?.t07Camera) frame.contentWindow.t07Camera.unlock();
    locked = null;
    frame.style.pointerEvents = '';
    fitReference();
    window.t07ReferenceInvalidate();
  };
  window.addEventListener('resize', fitReference);
  window.t07ReferenceInvalidate = () => {
    lastEvaluation = null;
    $('t07-psnr').textContent = 'Not evaluated';
    $('t07-reference-note').textContent = isCameraPathVisible() ? 'Turn off Show camera path before RGB reference evaluation.' : vehicleCleanupEnabled || vehicleCleanupHighlight ? 'Removal or highlight is active. Restore the original RGB map before evaluation.' : 'Enter a frame number and press Enter to align and evaluate. Full Viewer RGB map; frame scope is shown with the result.';
    $('t07-reference-note').removeAttribute('title');
  };
  window.t07ReferenceReady = () => {
    const [lo,hi] = config.sourceFrames;
    $('t07-reference-frame').placeholder = `Enter a frame number from ${lo} to ${hi}`;
    $('t07-reference-frame').setAttribute('title', `Source frame ${lo}–${hi}; press Enter to align and evaluate`);
  };
  async function capturePng(width, height, supersample) {
    const result = await bridge().captureFrame({width,height,supersample});
    const raw = Uint8ClampedArray.from(atob(result.data), c => c.charCodeAt(0));
    if (raw.length !== width*height*4) throw new Error('Unexpected rendered pixel count.');
    const canvas = document.createElement('canvas'); canvas.width=width; canvas.height=height;
    canvas.getContext('2d').putImageData(new ImageData(raw,width,height),0,0);
    const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/png'));
    if (!blob) throw new Error('PNG encoding failed.');
    return new Promise((resolve,reject) => {
      const reader=new FileReader(); reader.onload=()=>resolve(reader.result.split(',')[1]);
      reader.onerror=reject; reader.readAsDataURL(blob);
    });
  }
  async function evaluateReference() {
    if (isCameraPathVisible()) throw new Error('Turn off Show camera path before RGB reference evaluation.');
    if (vehicleCleanupEnabled || vehicleCleanupHighlight) throw new Error('Turn off highlight and moving vehicle removal before evaluating the original RGB map.');
    const ref = locked?.camera.reference;
    if (!ref || mode !== 'rgb') throw new Error('Enter a reference frame number in RGB Map and press Enter.');
    $('t07-psnr').textContent = 'Evaluating…';
    try {
      const supersample = Number($('t07-ss').value);
      const png = await capturePng(ref.width, ref.height, supersample);
      const meta = metadata();
      meta.render = {width:ref.width,height:ref.height,supersample,uiExcluded:true,renderer:bridge().app.graphicsDevice.deviceType};
      const result = await post('/api/evaluate',{frame:ref.frame,metadata:meta,png});
      lastEvaluation = {sceneId:config.sceneId,frame:ref.frame,variant,psnr_db:result.psnr_db,identical:result.identical,width:ref.width,height:ref.height,supersample,protocol:ref.protocol,path:result.path,scope:'Native reference-size diagnostic; export width may differ'};
      $('t07-psnr').textContent = result.identical ? '∞ dB' : `${result.psnr_db.toFixed(3)} dB`;
      $('t07-reference-note').textContent = `Frame ${ref.frame} · ${variant} · ${ref.width}×${ref.height} · ${supersample}× · ${ref.protocol}.`;
      $('t07-reference-note').title = `Evaluation receipt: ${result.path}`;
      status(`Reference frame ${ref.frame} aligned and evaluated. Click Unlock View to navigate.`);
    } catch(error) {
      lastEvaluation=null;
      $('t07-psnr').textContent='Evaluation failed';
      $('t07-reference-note').textContent=error.message;
      throw error;
    }
  }
  const referenceAction = action(evaluateReference);
  async function alignReferenceFrame(id, generation = mapGeneration) {
    const sceneId = config.sceneId;
    const response = await fetch(`/api/reference?scene=${encodeURIComponent(sceneId)}&frame=${id}`);
    const ref = await response.json();
    if (generation !== mapGeneration || mode !== 'rgb') return false;
    if (!response.ok) throw new Error(requestError(ref.error));
    if (ref.sceneId !== sceneId || ref.frame !== id) throw new Error('Reference frame does not match the request.');
    const camera = bridge().t07Camera.reference(poses.get(id), ref);
    locked = {schema:'t07-camera-v1', sceneId, aspect:ref.width/ref.height, camera};
    frame.style.pointerEvents = 'none';
    fitReference();
    $('t07-reference-frame').value = String(id);
    status(`Aligned to frame ${id}. Unlock View to navigate.`);
    return true;
  }
  window.t07AlignReferenceFrame = alignReferenceFrame;
  $('t07-reference-form').onsubmit = event => {
    event.preventDefault();
    if(busy) return;
    action(async () => {
      if (isCameraPathVisible()) throw new Error('Turn off Show camera path before RGB reference evaluation.');
      if (vehicleCleanupEnabled || vehicleCleanupHighlight) throw new Error('Turn off highlight and moving vehicle removal before evaluating the original RGB map.');
      window.t07ReferenceInvalidate();
      const text=$('t07-reference-frame').value.trim();
      if (!/^\d+$/.test(text)) throw new Error('Enter an integer frame number.');
      const id=Number(text), [lo,hi]=config.sourceFrames;
      if (!Number.isSafeInteger(id) || id<lo || id>hi || !poses.has(id)) throw new Error(`Enter a frame number from ${lo} to ${hi}.`);
      if(mode!=='rgb') throw new Error('Switch to RGB Map before evaluation.');
      if (await alignReferenceFrame(id)) await evaluateReference();
    })();
  };
  $('t07-ss').addEventListener('change',()=>{ if(locked?.camera.reference && mode==='rgb' && !isCameraPathVisible()) referenceAction(); });
  $('t07-export').onclick = action(async () => {
    const win = bridge();
    if (mode === 'sog') throw new Error('SOG uses progressive LOD. Switch to RGB Map to export the full point cloud.');
    const meta = metadata();
    win.t07Camera.lock(locked.camera);
    const width = 1920;
    const height = Math.round(width / locked.aspect);
    const supersample = Number($('t07-ss').value);
    if (width * height * supersample ** 2 > 16000000) throw new Error('The output is too large. Reduce antialiasing quality.');
    status(`Rendering ${width} × ${height} offscreen. Please wait…`);
    const result = await win.captureFrame({width, height, supersample});
    const raw = Uint8ClampedArray.from(atob(result.data), c => c.charCodeAt(0));
    if (raw.length !== width * height * 4) throw new Error('Unexpected rendered pixel count.');
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(raw, width, height), 0, 0);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('PNG encoding failed.');
    const png = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject; reader.readAsDataURL(blob);
    });
    meta.render = {width, height, supersample, engine: 'SuperSplat existing offscreen Capture.grab',
      renderer: win.app.graphicsDevice.deviceType, uiExcluded: true};
    const saved = await post('/api/export', {metadata: meta, png});
    status(`Saved to: ${saved.path} (PNG, camera JSON and SHA256).`);
  });
})();
