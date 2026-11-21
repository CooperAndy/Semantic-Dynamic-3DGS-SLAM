"""Create an offline human review page, separate from the map Demo."""
import argparse
import csv
import json
from pathlib import Path

HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>seq09 车辆轨迹核查</title>
<style>body{margin:20px auto;max-width:1280px;background:#111827;color:#e5e7eb;font:16px system-ui}button,select,input,textarea{font:inherit;padding:7px;margin:4px;border-radius:5px}select{max-width:95%}button{cursor:pointer}canvas{width:100%;background:#080c13}#crop{width:48%;height:200px;object-fit:contain}section{background:#1f2937;padding:14px;margin:12px 0;border-radius:9px}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.muted{color:#aab8cd}#time{width:60%}textarea{width:60%;vertical-align:middle}#prediction{white-space:pre-wrap}a{color:#93c5fd}</style>
<h1>seq09 完整轨迹核查</h1><p>这是实验核查页。最终地图展示仍是每辆车一个固定标签。先看原始连续影像，再填写真实状态；模型结果默认隐藏。</p>
<section><select id="track"></select><button id="prev">上一轨迹</button><button id="next">下一轨迹</button><span id="progress"></span><p id="description"></p>
<canvas id="view" width="1226" height="370"></canvas><div class="row"><button id="play">播放 / 暂停</button><button id="back">前一帧</button><button id="forward">后一帧</button><input type="range" id="time"><span id="frame"></span><label><input type="checkbox" id="box" checked>目标框</label></div>
<p class="muted">播放包括轨迹前后各 8 帧；缺少检测时继续显示原始影像。黄色框仅用于定位实例，并非真值。可关闭目标框检查背景相对运动。</p><canvas id="crop" width="500" height="240"></canvas></section>
<section><h2>人工核查记录</h2><p><label>真实车辆 ID <input id="object" placeholder="同一车辆跨 track 填同一个 ID"></label><label>身份关联 <select id="identity"><option value="">未核查</option><option value="yes">确认同一辆车</option><option value="no">存在误关联 / 需拆分</option><option value="uncertain">无法确认</option></select></label></p>
<p><label>真实状态 <select id="truth"><option value="">未标注</option><option value="static">静态</option><option value="dynamic">动态</option><option value="uncertain">无法判断</option><option value="state_change">真实状态变化</option></select></label><label>核查人 <input id="reviewer"></label><label>依据帧 <input id="evidence" placeholder="例如 398–433"></label></p>
<p>关联说明 <input id="identityNotes" size="55" placeholder="疑似断裂、同车其他 track、误关联时间"></p><p>备注 <textarea id="notes" rows="2"></textarea><button id="save">保存当前</button></p>
<p><button id="export">导出标注 CSV</button><label>导入已有标注 CSV <input type="file" id="import" accept=".csv"></label></p><p class="muted">记录自动保存在此浏览器的本地存储中，仍请及时导出备份。未标注不会被当作静态；模型输出不能充当真值。</p></section>
<section><label><input type="checkbox" id="showPred">显示模型结果和固定标签对照</label><select id="config"></select><pre id="prediction" hidden></pre><p>这些配置都是探索候选。尚无人工真值时，不能按 uncertain 最少选择“最准”的配置。</p></section>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let index=0,frame=300,timer=null,renderToken=0;
const keys=Object.keys(DATA.tracks).sort((a,b)=>Number(a.split(':')[0])-Number(b.split(':')[0]));const storeKey='seq09-fixed-labels-v1-'+DATA.mapSha;let labels={};try{labels=JSON.parse(localStorage.getItem(storeKey)||'{}')}catch(e){}
const fields=['track_segment_id','true_object_id','true_state','identity_ok','split_or_merge_notes','reviewer','evidence_frames','notes'];
const inputs={true_object_id:'object',true_state:'truth',identity_ok:'identity',split_or_merge_notes:'identityNotes',reviewer:'reviewer',evidence_frames:'evidence',notes:'notes'};
for(const key of keys){let o=document.createElement('option');let rows=DATA.tracks[key];o.value=key;o.textContent=`track ${key} · ${rows.length} 次检测 · ${rows[0].source_frame}–${rows.at(-1).source_frame}`;$('track').appendChild(o)}
for(const c of DATA.configs){let o=document.createElement('option');o.value=c;o.textContent=c;$('config').appendChild(o)}$('config').value='strict8_z0.5-40_far20';
function current(){return keys[index]}function persist(){try{localStorage.setItem(storeKey,JSON.stringify(labels))}catch(e){alert('本地存储失败，请立即导出 CSV')}}
function save(){let row={track_segment_id:current()};for(const [key,id] of Object.entries(inputs))row[key]=$(id).value;labels[current()]=row;persist();progress()}
function progress(){$('progress').textContent=`${index+1}/${keys.length} · 已填写 ${Object.values(labels).filter(r=>r.true_state).length} 条`}
function choose(i){if(timer){clearInterval(timer);timer=null}index=Math.max(0,Math.min(keys.length-1,i));$('track').value=current();let rows=DATA.tracks[current()];$('time').min=Math.max(300,rows[0].source_frame-8);$('time').max=Math.min(499,rows.at(-1).source_frame+8);frame=Number($('time').min);for(const [key,id]of Object.entries(inputs))$(id).value=labels[current()]?.[key]||'';$('description').textContent=`轨迹段 ${current()}，${rows.length} 次检测。请核查是否始终为同一辆车，并查看完整运动过程。`;progress();draw();prediction()}
function draw(){frame=Math.max(Number($('time').min),Math.min(Number($('time').max),frame));$('time').value=frame;$('frame').textContent=`原始帧 ${frame}`;const token=++renderToken,img=new Image();img.onload=()=>{if(token!==renderToken)return;const ctx=$('view').getContext('2d');ctx.drawImage(img,0,0,1226,370);const row=DATA.tracks[current()].find(r=>r.source_frame===frame);const crop=$('crop').getContext('2d');crop.clearRect(0,0,500,240);if(row){let[x1,y1,x2,y2]=row.bbox;if($('box').checked){ctx.strokeStyle='#facc15';ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1)}let pad=25,sx=Math.max(0,x1-pad),sy=Math.max(0,y1-pad),w=Math.min(1226-sx,x2-x1+2*pad),h=Math.min(370-sy,y2-y1+2*pad),scale=Math.min(500/w,240/h);crop.drawImage(img,sx,sy,w,h,(500-w*scale)/2,(240-h*scale)/2,w*scale,h*scale)}};img.onerror=()=>{if(token===renderToken)$('frame').textContent+= ' · 原始图片未找到'};img.src=`review_assets/rgb/${String(frame).padStart(6,'0')}.jpg`}
function prediction(){let p=DATA.candidates.find(r=>r.config===$('config').value&&r.track_segment_id===current());$('prediction').hidden=!$('showPred').checked;$('prediction').textContent=JSON.stringify(p,null,2)}
$('track').onchange=()=>{save();choose(keys.indexOf($('track').value))};$('prev').onclick=()=>{save();choose(index-1)};$('next').onclick=()=>{save();choose(index+1)};$('save').onclick=save;$('time').oninput=()=>{frame=Number($('time').value);draw()};$('back').onclick=()=>{frame--;draw()};$('forward').onclick=()=>{frame++;draw()};$('box').onchange=draw;$('config').onchange=prediction;$('showPred').onchange=prediction;
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{frame=frame>=Number($('time').max)?Number($('time').min):frame+1;draw()},125)};
function quote(v){return '"'+String(v??'').replaceAll('"','""')+'"'}$('export').onclick=()=>{save();const lines=[fields.map(quote).join(','),...keys.map(k=>fields.map(f=>quote((labels[k]||{track_segment_id:k})[f]||'')).join(','))];const link=document.createElement('a');let url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));link.href=url;link.download='seq09_human_labels.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
function parseCSV(text){const out=[];let row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){let c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++}else quoted=!quoted}else if(c===','&&!quoted){row.push(cell);cell=''}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(Boolean))out.push(row);row=[];cell=''}else cell+=c}if(cell||row.length){row.push(cell);out.push(row)}return out}
$('import').onchange=async()=>{let f=$('import').files[0];if(!f)return;let table=parseCSV((await f.text()).replace(/^\ufeff/,'')),head=table.shift();if(!fields.every(k=>head.includes(k))){alert('CSV 列不完整');return}for(let cells of table){let r=Object.fromEntries(head.map((h,i)=>[h,cells[i]||'']));if(keys.includes(r.track_segment_id))labels[r.track_segment_id]=r}persist();choose(index)};
window.addEventListener('beforeunload',save);choose(0);</script></html>'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args(); run = Path(a.run)
    rows = json.loads((run/'observations.json').read_text(encoding='utf-8'))
    tracks = {}
    for row in rows: tracks.setdefault(row['track_segment_id'], []).append(row)
    candidates = list(csv.DictReader((run/'candidate_tracks.csv').open(encoding='utf-8-sig')))
    manifest = json.loads((run/'input_manifest.json').read_text(encoding='utf-8'))
    data = dict(tracks=tracks, candidates=candidates, configs=list(dict.fromkeys(r['config'] for r in candidates)), mapSha=manifest['source_map_sha256'])
    encoded = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    (run/'review.html').write_text(HTML.replace('__DATA__', encoded), encoding='utf-8')
    print(run/'review.html')


if __name__ == '__main__': main()
