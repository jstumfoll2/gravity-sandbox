"""Headless preview: runs step.comp/holes.comp and the render shaders on a desktop GL
context (ARB_ES3_1_compatibility), prints stats and writes a PNG. Mirrors SimRenderer.kt.
Usage: python3 tools/preview.py --n 4096 --steps 300 --preset galaxies --out preview.png
"""
import moderngl, numpy as np, sys, time
from PIL import Image
import os
S=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','app','src','main','assets','shaders','')
src=lambda n: open(S+n).read()
import argparse
ap=argparse.ArgumentParser(description='Run the real compute and render shaders headless (Mesa llvmpipe) and save a PNG.')
ap.add_argument('--n',type=int,default=4096); ap.add_argument('--steps',type=int,default=300)
ap.add_argument('--preset',default='galaxies',choices=['disk','galaxies','cloud']); ap.add_argument('--out',default='preview.png')
ap.add_argument('--heat',type=float,default=.02)
A=ap.parse_args(); N=A.n; STEPS=A.steps; preset=A.preset; out=A.out
ctx=moderngl.create_standalone_context(backend='egl', require=430)
step=ctx.compute_shader(src('step.comp')); holes=ctx.compute_shader(src('holes.comp'))
part=ctx.program(vertex_shader=src('particle.vert'), fragment_shader=src('particle.frag'))
fade=ctx.program(vertex_shader=src('fullscreen.vert'), fragment_shader=src('fade.frag'))
pres=ctx.program(vertex_shader=src('fullscreen.vert'), fragment_shader=src('present.frag'))
hprog=ctx.program(vertex_shader=src('hole.vert'), fragment_shader=src('hole.frag'))
G=40000.; rng=np.random.default_rng(1)
P=np.zeros((N,8),'f4'); P[:,0:2]=1e6
n=int(N*0.7); m=1/(N*0.7)
H=np.zeros((16,8),'f4'); H[:,0:2]=1e6
def hole(i,x,y,vx,vy,M): H[i]=[x,y,M,max(1.5,6*M),vx,vy,0,0]
def disk(s,c,cx,cy,cvx,cvy,rin,rout,Mc,dm,d):
    u=rng.random(c); r=rin+(rout-rin)*u; a=rng.random(c)*2*np.pi
    vc=np.sqrt(G*(Mc+dm*u)/r)*d
    P[s:s+c,0]=cx+r*np.cos(a); P[s:s+c,1]=cy+r*np.sin(a); P[s:s+c,2]=m; P[s:s+c,3]=1
    P[s:s+c,4]=cvx-np.sin(a)*vc+rng.normal(0,.03,c)*vc; P[s:s+c,5]=cvy+np.cos(a)*vc+rng.normal(0,.03,c)*vc
nh=0
if preset=='disk': hole(0,0,0,0,0,1.0); nh=1; disk(0,n,0,0,0,0,12,85,1.0,1.0,1)
elif preset=='galaxies':
    hole(0,-60,-25,9,3,.5); hole(1,60,25,-9,-3,.5); nh=2
    disk(0,n//2,-60,-25,9,3,5,40,.5,.5,1); disk(n//2,n-n//2,60,25,-9,-3,5,40,.5,.5,-1)
elif preset=='cloud':
    r=70*np.sqrt(rng.random(n)); a=rng.random(n)*2*np.pi; vc=np.sqrt(G*(r/70)**2/(r+1))*.3
    P[:n,0]=r*np.cos(a);P[:n,1]=r*np.sin(a);P[:n,2]=m;P[:n,3]=1;P[:n,4]=-np.sin(a)*vc;P[:n,5]=np.cos(a)*vc
B=[ctx.buffer(P.tobytes()),ctx.buffer(P.tobytes())]; HB=ctx.buffer(H.tobytes()); AB=ctx.buffer(np.zeros(48,'i4').tobytes())
U=dict(uCount=N,uHoleCount=nh,uDt=1/120,uG=G,uSoft2=1.0,uH=1.5,uK=30.,uVisc=2.,uDrag=0.,uHoleSoft2=.25,uBound=600.,uHeatGain=A.heat,uHeatDecay=.6,uTouchCount=0)
for k,v in U.items():
    if k in step: step[k].value=v
for k,v in dict(uCount=N,uHoleCount=nh,uDt=1/120,uG=G,uHoleSoft2=.25,uRadiusPerMass=6.,uMinRadius=1.5,uBound=600.).items():
    if k in holes: holes[k].value=v
W,Hh=540,1100
tex=ctx.texture((W,Hh),4,dtype='f2'); fbo=ctx.framebuffer([tex])
ctx.enable(moderngl.PROGRAM_POINT_SIZE)
half=100.; sx=1/half; sy=1/(half*Hh/W)
empty=ctx.vertex_array(fade,[]); emptyp=ctx.vertex_array(pres,[])
cur=0; t=time.time()
def render():
    fbo.use(); ctx.enable(moderngl.BLEND)
    ctx.blend_func=(moderngl.ZERO,moderngl.SRC_ALPHA); fade['uKeep'].value=.55; empty.render(moderngl.TRIANGLES,3)
    ctx.blend_func=(moderngl.ONE,moderngl.ONE)
    for k,v in dict(uScale=(sx,sy),uCenter=(0,0),uPointPx=3.,uDensGlow=.08,uHeatScale=.05,uBrightness=.35).items():
        if k in part: part[k].value=v
    ctx.vertex_array(part,[(B[cur],'4f 4f','aPos','aVel')]).render(moderngl.POINTS,N)
def present(path):
    fb=ctx.simple_framebuffer((W,Hh)); fb.use(); fb.clear(); ctx.disable(moderngl.BLEND)
    tex.use(0); pres['uTex'].value=0; pres['uExposure'].value=1.5; emptyp.render(moderngl.TRIANGLES,3)
    ctx.enable(moderngl.BLEND); ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
    for k,v in dict(uScale=(sx,sy),uCenter=(0,0),uPxPerUnit=W/200).items():
        if k in hprog: hprog[k].value=v
    ctx.vertex_array(hprog,[(HB,'4f 4f','aPos','aVel')]).render(moderngl.POINTS,16)
    Image.frombytes('RGB',(W,Hh),fb.read()).transpose(Image.FLIP_TOP_BOTTOM).save(path)
for s in range(STEPS):
    B[cur].bind_to_storage_buffer(0); B[1-cur].bind_to_storage_buffer(1); HB.bind_to_storage_buffer(2); AB.bind_to_storage_buffer(3)
    step.run(N//256); ctx.memory_barrier()
    if nh:
        B[1-cur].bind_to_storage_buffer(0); holes.run(1); ctx.memory_barrier()
    cur=1-cur
    if s%2==1: render()
    if s%100==0 or s==STEPS-1:
        p=np.frombuffer(B[cur].read(),'f4').reshape(N,8); al=p[:,3]>.5; h=np.frombuffer(HB.read(),'f4').reshape(16,8)
        print(s,'alive',al.sum(),'dens>2 %.2f'%(p[al,7]>2).mean(),'maxdens %.1f'%p[al,7].max(),'heat p50 %.2f p99 %.2f'%tuple(np.percentile(p[al,6],[50,99])),'holes M',np.round(h[:nh,2],3),'%.1fs'%(time.time()-t),flush=True)
present(out)
