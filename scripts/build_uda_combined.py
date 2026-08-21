import os,glob,argparse
ap=argparse.ArgumentParser(); ap.add_argument("--loso_dir",required=True); ap.add_argument("--pseudo",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
def link(srcs,od):
    os.makedirs(od,exist_ok=True); n=0
    for root in srcs:
        for sc in sorted(glob.glob(os.path.join(root,"*"))):
            dst=os.path.join(od,os.path.basename(sc))
            if os.path.islink(dst): os.remove(dst)
            if not os.path.exists(dst): os.symlink(os.path.abspath(sc),dst); n+=1
    return n
tr=link([f"{a.loso_dir}/train",a.pseudo],f"{a.out}/train"); va=link([f"{a.loso_dir}/val"],f"{a.out}/val")
print(f">> {a.out}: train {tr} / val {va}")
