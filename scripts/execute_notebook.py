"""Execute every plain-Python notebook cell locally and persist real outputs.
Colab-only Drive/upload/download branches are skipped by the notebook itself.
Usage: VISION_DATASET_ZIP=/path/data.zip python scripts/execute_notebook.py
"""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import base64
import io
import json
import sys
import traceback

root=Path(__file__).resolve().parents[1]
path=root/'notebooks/Project03_VISION_HE.ipynb'
notebook=json.loads(path.read_text())
namespace={'__name__':'__notebook__'}
from IPython.core.interactiveshell import InteractiveShell
shell=InteractiveShell.instance()
outputs=[]
def publish(data, metadata=None, **kwargs):
    data = {mime: base64.b64encode(value).decode('ascii') if isinstance(value, bytes) else value for mime, value in data.items()}
    outputs.append({'output_type':'display_data','data':data,'metadata':metadata or {}})
shell.display_pub.publish=publish
class Tee(io.StringIO):
    def write(self,s):
        sys.__stdout__.write(s);sys.__stdout__.flush()
        if outputs and outputs[-1].get('output_type')=='stream':outputs[-1]['text']+=s
        else:outputs.append({'output_type':'stream','name':'stdout','text':s})
        return len(s)

count=0
for cell in notebook['cells']:
    if cell['cell_type']!='code':continue
    count+=1
    if '--resume-evaluation' in sys.argv and 'metadata = train(' in ''.join(cell['source']):
        assert cell['execution_count'] is not None, 'No completed training cell to resume.'
        _, metadata = namespace['load_bundle'](root/'models')
        namespace.update(RUN=root, metadata=metadata)
        print('Keeping genuine completed training outputs; resuming evaluation/export.')
        continue
    outputs=[];cell['execution_count']=count;cell['outputs']=outputs
    print(f'Executing notebook cell {count}',flush=True)
    try:
        with redirect_stdout(Tee()),redirect_stderr(Tee()):
            exec(compile(''.join(cell['source']),f'{path.name}:cell{count}','exec'),namespace)
    except Exception as exc:
        outputs.append({'output_type':'error','ename':type(exc).__name__,'evalue':str(exc),'traceback':traceback.format_exc().splitlines()})
        path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
        raise
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
print('All notebook cells executed successfully.')
