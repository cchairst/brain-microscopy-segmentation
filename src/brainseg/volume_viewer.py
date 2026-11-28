"""Write a standalone slice viewer for a finished CMC volume."""

import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image


def write_viewer(path, subject, axis, ref, tissue, source, reason, priority):
    valid = ref[ref > 0]
    lo, hi = np.percentile(valid, [1, 99]) if valid.size else (0, 1)
    gray = (255 * np.clip((ref - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
    # Blue values 0–2 retain the normal source IDs; 3–6 encode the four unresolved states.
    display_source = np.where(source == 255, 2 + reason, source).astype(np.uint8)
    frames = []
    for layers in zip(*(np.moveaxis(a, axis, 0) for a in (gray, tissue, display_source))):
        # Each lossless PNG stores intensity and the two categorical IDs.
        pixels = np.stack([np.flipud(a.T) for a in layers], axis=-1)
        buf = BytesIO()
        Image.fromarray(pixels).save(buf, format='PNG')
        frames.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    html = TEMPLATE.replace('SUBJECT', subject).replace('PLANE', 'Coronal' if axis == 1 else 'Sagittal')
    path.write_text(
        html.replace('PRIORITY_INDEX', str(priority)).replace('FRAMES', json.dumps(frames)), encoding='utf-8'
    )


TEMPLATE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SUBJECT · Whole-brain segmentation</title>
<style>
*{box-sizing:border-box}body{background:#111922;color:#ebf0f1;font:16px/1.45 system-ui,sans-serif;
max-width:740px;margin:0 auto;padding:20px}h1{font-size:1.45rem;margin:0}p{margin:6px 0 16px}
canvas{display:block;width:100%;background:#070b11;image-rendering:pixelated}
label,output{display:inline-block}input[type=range]{width:100%;margin:12px 0;accent-color:#83d3c3}
select,button{padding:8px;font:inherit;background:#202d39;color:inherit;border:1px solid #718391;border-radius:4px}
.controls{display:flex;gap:8px;align-items:center;justify-content:space-between;flex-wrap:wrap}
#legend{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}.chip{display:inline-block;width:12px;height:12px;
margin-right:5px;border-radius:2px}
</style><main><h1>SUBJECT · Whole-brain segmentation</h1><p>PLANE slices on the ex vivo MRI grid</p>
<label>View <select id="mode"><option value="overlay">Tissue overlay</option><option value="tissue">Tissue mask</option>
<option value="why">Why uncertain?</option><option value="source">Label source</option>
<option value="raw">PS-OCT reflectivity</option></select></label>
<button id="priority">Review priority slice</button>
<div id="legend"></div><canvas id="scan" aria-label="Selected brain slice"></canvas>
<input id="slice" aria-label="MRI-grid slice" type="range" min="0" step="1">
<div class="controls"><button id="prev">Previous</button><output id="position"></output>
<button id="next">Next</button></div>
<p id="reasonHelp" hidden>The signal gate checks for positive retardance and reflectivity above the floor.
At the Otsu gate, signal passed,
but the optical segmentation gave no gray/white label. MRI boundary: at least one neighbouring voxel differs.</p>
<p>MRI fills require the same tissue label at the six adjacent MRI voxels. Uncertain regions have ID 255.</p></main>
<script>
const frames=FRAMES;
const reasons={1:['Signal gate + MRI unknown',[228,85,91]],2:['Signal gate + MRI boundary',[247,172,65]],
3:['Otsu gate + MRI unknown',[183,126,217]],4:['Otsu gate + MRI boundary',[88,175,211]]};
const get=id=>document.getElementById(id);
const tissue={1:['Gray matter',[241,166,77]],2:['White matter',[92,167,241]],3:['Possible fluid',[183,133,216]],
255:['Uncertain',[128,130,130]]};
const source={1:['Optical',[222,226,232]],2:['MRI fill',[42,182,155]],255:['Uncertain',[128,130,130]]};
const slider=get('slice'), mode=get('mode'), canvas=get('scan');
const context=canvas.getContext('2d',{willReadFrequently:true});
slider.max=frames.length-1;slider.value=Math.floor(frames.length/2);let revision=0;
function render(){
  const index=Number(slider.value),view=mode.value,id=++revision;
  const table=view==='why'?reasons:view==='source'?source:tissue;
  get('reasonHelp').hidden=view!=='why';
  get('position').textContent='Slice '+(index+1)+' of '+frames.length+' · index '+index;
  get('prev').disabled=index===0;get('next').disabled=index===frames.length-1;
  get('legend').replaceChildren();
  if(view!=='raw')for(const [label,color] of Object.values(table)){
    const item=document.createElement('span'),chip=document.createElement('span');
    chip.className='chip';chip.style.background='rgb('+color.join(',')+')';
    item.append(chip,label);get('legend').append(item);
  }
  const img=new Image();img.onload=()=>{
    if(id!==revision)return;
    canvas.width=img.width;canvas.height=img.height;context.drawImage(img,0,0);
    const pixels=context.getImageData(0,0,img.width,img.height),data=pixels.data;
    for(let i=0;i<data.length;i+=4){
      const gray=data[i],code=data[i+2],unresolved=code>=3;
      const key=view==='why'?(unresolved?code-2:0):view==='source'?(unresolved?255:code):data[i+1];
      const entry=table[key];
      for(let k=0;k<3;k++){
        if(view==='raw')data[i+k]=gray;
        else if(view==='overlay')data[i+k]=entry ? .45*gray+.55*entry[1][k] : gray;
        else if(view==='why' && !entry)data[i+k]=.28*gray;
        else data[i+k]=entry ? entry[1][k] : 12;
      }
    }
    context.putImageData(pixels,0,0);
  };img.src='data:image/png;base64,'+frames[index];
}
get('priority').onclick=()=>{slider.value=PRIORITY_INDEX;mode.value='why';render()};
slider.oninput=mode.onchange=render;
get('prev').onclick=()=>{slider.stepDown();render()};get('next').onclick=()=>{slider.stepUp();render()};render();
</script></html>"""
