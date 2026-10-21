"""Inspect the actual ZIP, exclude incorrect_mask, deduplicate and make disjoint splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from backend.app.preprocessing import decode_image, image_array, InvalidImage
from ml import CLASS_MAPPING
from ml.artifacts import sha256, write_json

SOURCE={'url':'https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset',
        'slug':'shiekhburhan/face-mask-dataset','version':1,'license':'CC0: Public Domain (Kaggle metadata)'}


def prepare(archive, output):
    archive, output=Path(archive),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    images=[];records=[];seen={};conflicts=set();duplicates=[];invalid=[]
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if not n.endswith('/'))
        folders=Counter(str(PurePosixPath(n).parent) for n in names)
        print('Actual archive structure:',dict(folders),flush=True)
        expected={f'FMD_DATASET/{label}/{kind}' for label in CLASS_MAPPING for kind in ['simple','complex']}
        if not expected.issubset(folders):raise ValueError('Unexpected dataset structure; inspect before training.')
        selected=[n for n in names if str(PurePosixPath(n).parent) in expected]
        for i,name in enumerate(selected):
            parts=PurePosixPath(name).parts;label=parts[1]
            try:
                image=decode_image(z.read(name))
                x=np.rint(image_array(image)*255).astype(np.uint8)
                digest=hashlib.sha256(x.tobytes()).hexdigest()
                if digest in seen:
                    prior=seen[digest]
                    if records[prior]['label']!=label:conflicts.add(prior)
                    duplicates.append({'path':name,'same_as':records[prior]['path'],'conflicting_label':records[prior]['label']!=label})
                    continue
                # Exact 64-bit difference-hash groups are kept together. This is not face recognition.
                gray=np.asarray(image.resize((9,8),Image.Resampling.BILINEAR).convert('L'))
                group=np.packbits(gray[:,1:]>gray[:,:-1]).tobytes().hex()
                seen[digest]=len(records)
                records.append({'path':name,'label':label,'subgroup':parts[2], 'sha256_pixels':digest,'visual_group':group})
                images.append(x)
            except InvalidImage as exc:invalid.append({'path':name,'error':str(exc)})
            if (i+1)%1000==0:print('Decoded',i+1,'/',len(selected),flush=True)
        keep=[i for i in range(len(records)) if i not in conflicts]
        x=np.stack([images[i] for i in keep]);records=[records[i] for i in keep]
        y=np.array([CLASS_MAPPING[r['label']] for r in records],dtype=np.int32)
        groups=np.array([r['visual_group'] for r in records])
        rest,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=42).split(x,y,groups))
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=43).split(x[rest],y[rest],groups[rest]))
        splits={'train':rest[tr],'validation':rest[va],'test':test}
        sets=[set(groups[ix]) for ix in splits.values()]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        arrays={};counts={}
        for split,ix in splits.items():
            assert set(y[ix])=={0,1}
            arrays['x_'+split]=x[ix];arrays['y_'+split]=y[ix]
            counts[split]=dict(Counter(records[i]['label'] for i in ix))
            for i in ix:records[i]['split']=split
        np.savez_compressed(output/'images.npz',**arrays)
        manifest={'source':SOURCE,'archive_sha256':sha256(archive),'actual_structure':dict(folders),
                  'selected_before_cleaning':len(selected),'excluded_incorrect_mask':sum(v for k,v in folders.items() if '/incorrect_mask/' in k),
                  'duplicate_files_removed':len(duplicates),'conflicting_representatives_removed':len(conflicts),
                  'invalid_images':invalid,'duplicates':duplicates,'usable_images':len(records),
                  'split_method':'GroupShuffleSplit; 70/15/15 approximate; exact dHash groups; seeds 42 and 43',
                  'counts':counts,'identity_disjoint':'Not guaranteed: dataset has no person IDs.', 'records':records}
        write_json(output/'manifest.json',manifest)
        # Fixed first-by-path held-out samples, not cherry-picked by model success.
        fixture_dir=output/'test_samples';fixture_dir.mkdir(exist_ok=True)
        fixture_sources={}
        for label in CLASS_MAPPING:
            record=next(r for r in records if r['split']=='test' and r['label']==label)
            (fixture_dir/f'{label}.jpg').write_bytes(z.read(record['path']))
            fixture_sources[f'{label}.jpg']=record['path']
        write_json(fixture_dir/'sources.json',fixture_sources)
    print('Prepared counts:',counts,'duplicates:',len(duplicates),'invalid:',len(invalid),flush=True)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('--output',default='data/prepared')
    args=parser.parse_args();prepare(args.archive,args.output)
