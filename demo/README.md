# KITTI Dynamic Semantic Demo

This source release includes the viewer and scene profiles. Generate the scene assets first and point the selected profile to the local map, observations, and reference images.

Double-click **Open Demo.cmd** to start the local Viewer. Existing launch shortcuts continue to work. Python 3 is required. The launcher exits after startup; the local service runs in the background and repeated launches reuse it. After the last Demo tab closes, the service stops after about ten seconds. A crashed browser may take approximately three minutes to expire its heartbeat lease. The server is not stopped by refreshing a page or closing one of several active Demo tabs.

## Display

Select a sequence and frame range at the top of the sidebar. The existing F200/F1 scenes use the F1 appearance-refined map by default; their F200 source maps remain stored. Display modes are RGB Map, streamed SOG/LOD where available, Semantic Map and Vehicle States. In RGB Map, use the sidebar's **Show camera path** option to add camera markers while keeping the current view; Camera Path is no longer a separate tab.

Supported scenes use an independent cached Gaussian path layer. The first enable loads only its small path PLY (about 112 KB for seq09, with sizes varying by scene); later toggles show/hide the cached layer without replacing the original RGB map or iframe. Cache lifetime is the current RGB Viewer; leaving that view or changing scenes releases it. PNG captures include visible markers, while reference PSNR excludes them. The original combined RGB-with-trajectory files are preserved; append `&path-layer=legacy` to a scene URL for the previous file-switching behavior. Marker shapes, sizes and close-camera occlusion are unchanged.

Map Information contains provenance and the scope of each scene. KITTI09 and KITTI17 use ORB-SLAM3 estimated poses. The F1 refinement does not improve camera poses or geometry: it optimizes color and opacity with fixed geometry and cameras.

Vehicle states are offline predictions. KITTI17 provides frame selection; the first seven frames lack a full in-range history and remain unknown. Seq 09 Conf 0.50 S42 uses fixed offline object states. The labels use estimated 3D positions from masks and depth; object-level Gaussian binding and occlusion are unverified. Predictions have not been evaluated for accuracy in this Demo.

Vehicle Display is available only in Vehicle States. Remove Predicted Moving Vehicles uses a reversible display copy. Preview Removal marks selected points in red without hiding them. Leaving Vehicle States resets both controls, so removal and highlight never carry over into RGB, semantic or SOG/LOD views. Neither action edits the original map. Semantic class and motion state are different concepts.

English reference images use the existing prediction boxes, IDs, states and scores, rendered over the original RGB frames. Click a reference image to enlarge it; scroll to zoom, drag to pan, and use Esc or Close to dismiss it. Reset view restores the initial framing. Original prediction JSON and original reference images are preserved.

In Semantic Map, the sidebar shows only labels whose colors occur in the entire semantic Gaussian map, independent of the current view. The inventory is generated from the PLY colors, including Unknown; a small number of Gaussians can make a label appear even when it is hard to see. Blending and opacity can change their appearance on the rendered map. After adding or replacing semantic maps, run `python 源码/build_semantic_legend.py` to refresh the inventory.

## Camera and export

- **Reload map** (circular arrow beside the Gaussian count): reload the selected scene and current display mode, restoring its initial view and resetting temporary display and evaluation settings.

- **Lock View / Unlock View:** freeze the camera or resume navigation.
- **Save PNG:** save at 1920-pixel width, preserving the captured aspect ratio. The camera is locked automatically if needed. Each new capture folder contains `render.png`, `camera.json` and `SHA256.json`. Folder names use local date, sequence and frame range, followed by one counter shared by all captures that day: `20261112(seq07， 850-1049)_001`, then `20261112(seq09， 300-499)_002`. A new date starts at `_001`.
- **Restore Camera:** choose a capture from this Demo's current export folder, newest first and filtered to the exact current scene and source-frame range. The option text is exactly the folder name; capture dates and display-mode labels are not appended. A preview helps identify the saved view. Use **Choose JSON File…** to browse camera files in a local Windows picker, which always starts in this Demo's export folder. The local Python runtime requires Tkinter for this picker. This restores the camera only; select the desired display mode and state frame separately.
- **Export Settings:** choose 1x sampling or 2x supersampling. This affects exported images and reference PSNR renders, not continuous on-screen map rendering.

PNG exports contain the Gaussian render. HTML vehicle-state labels and sidebar controls are not included. Existing output directories and project paths keep their original names for compatibility.

## Reference evaluation

The **Reference Evaluation / PSNR** panel is always expanded near the top of the RGB Map sidebar. Enter an original frame number and press Enter: the Viewer aligns and locks to that frame, renders the reference-sized image, then displays its PSNR. The panel is absent from other modes. It stays visible when **Show camera path** is on, but its input is disabled with a reminder to turn off markers before scoring. The input also waits until the map is ready and an evaluation has finished. Camera markers, vehicle removal and highlight are excluded from scoring. Evaluation uses the original reference-image dimensions, regardless of the 1920-pixel export width. Changing antialiasing re-evaluates an aligned RGB reference view. These are seen-view Viewer diagnostics, not independent-test results.

Reference evaluation depends on configured original images and calibration files. Running evaluation on a different computer has not been verified. Map and reference-display assets are stored locally using package-relative paths.

## Navigation and troubleshooting

Left drag orbits, right drag pans, the scroll wheel zooms, WASD moves, and double-click sets the orbit center. The map's help button opens Navigation Guide. X is initial forward, Y initial right and Z up.

Refresh an existing Viewer page after an interface update. If the service has exited, launch the Demo again. Runtime logs remain in the existing logs directory; startup errors are written to `startup_error.log`. For foreground debugging, use `python launch_t07.py --no-browser` and stop with Ctrl+C.
