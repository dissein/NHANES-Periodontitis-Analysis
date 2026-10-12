from pathlib import Path
import subprocess,sys

def main():
    source=Path(__file__).resolve().parent
    for name in ['build_multicycle.py','study_multicycle.py','supplementary_multicycle.py','age_baseline.py','refit_uncertainty.py','test_multicycle.py','finalize_multicycle.py','verify_delivery.py']:
        print('Running',name,flush=True)
        subprocess.run([sys.executable,'-X','utf8',str(source/name)],check=True,cwd=source.parent)

if __name__=='__main__':main()
