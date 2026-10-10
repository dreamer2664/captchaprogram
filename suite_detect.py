"""Deterministic detection suites: real-page clone + bare cards + old composite."""
import random, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import finder, realpage
from generator import generate

def run(N=100):
    det=tb=0; lat=[]; bad=[]
    for seed in range(N):
        random.seed(1000+seed)
        dual = seed%2==1
        page, meta = realpage.render_page(dual=dual, seed=seed, meta=True)
        t0=time.time(); boxes, tbx = finder.detect_all(page); lat.append(time.time()-t0)
        ious=[max(finder.box_iou(b,r) for r in meta["boxes"]) for b in boxes] if boxes else [0]
        ok = len(boxes)==len(meta["boxes"]) and min(ious)>0.55
        det+=ok
        t=meta["textbox"]
        tb += tbx is not None and abs(tbx[0]-(t[0]+t[2]//2))<40 and abs(tbx[1]-(t[1]+t[3]//2))<20
        if not ok and len(bad)<5: bad.append((seed,dual,[(int(v) for v in b) and tuple(map(int,b)) for b in boxes],meta["boxes"]))
    print(f"real-page: detect {det}/{N}  textbox {tb}/{N}  mean {np.mean(lat)*1000:.0f}ms max {np.max(lat)*1000:.0f}ms")
    for b in bad: print("   bad:", b)
    ok2=0; bad2=[]
    for seed in range(60):
        random.seed(500+seed)
        cap,truth = generate()
        page = Image.new("RGB",(700,430),(255,255,255))
        page.paste(cap,(60,120))
        boxes = finder.find_captcha_boxes(page)
        good = len(boxes)==1 and finder.box_iou(boxes[0],(60,120,cap.width,cap.height))>0.6
        ok2+=good
        if not good and len(bad2)<3: bad2.append((seed,boxes,truth))
    print(f"bare-card: {ok2}/60")
    for b in bad2: print("   bad:", b)
    # old-style composite (grey panels, grey buttons, no red ring)
    F="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ok3=0
    for seed in range(40):
        random.seed(7700+seed)
        rng=random.Random(7700+seed)
        dual = seed%2==1
        page=Image.new("RGB",(760,430),(250,250,252)); d=ImageDraw.Draw(page)
        d.text((30,20),"CAPTCHA TEST",font=ImageFont.truetype(F,20),fill=(30,30,30))
        d.rectangle([30,52,30+rng.randint(100,340),62],fill=(200,40,40))
        d.rectangle([30,352,340,392],outline=(120,120,125),width=2,fill=(255,255,255))
        d.rectangle([360,352,540,392],fill=(52,88,190))
        rects=[]; x=rng.randint(30,80); y=rng.randint(110,200)
        for i in range(2 if dual else 1):
            cap,_=generate(); page.paste(cap,(x,y)); rects.append((x,y,cap.width,cap.height))
            x+=cap.width+rng.randint(14,60)
        boxes=finder.find_captcha_boxes(page)
        ious=[max(finder.box_iou(b,r) for r in rects) for b in boxes] if boxes else [0]
        ok3 += len(boxes)==len(rects) and min(ious)>0.55
    print(f"old-composite (no red ring): {ok3}/40")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv)>1 else 100)
