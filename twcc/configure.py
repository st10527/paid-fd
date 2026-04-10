#!/usr/bin/env python3
"""
TWCC 快速設定檢查 + 修改計畫 ID
================================
在本機執行，自動修改所有 SLURM script 的 PROJECT_ID

Usage:
    python twcc/configure.py GOV112345
    python twcc/configure.py MST108XXX --email your@email.com
"""

import sys
import os
import re
import glob


def main():
    if len(sys.argv) < 2:
        print("Usage: python twcc/configure.py <PROJECT_ID> [--email <EMAIL>]")
        print("Example: python twcc/configure.py GOV112345 --email student@tku.edu.tw")
        sys.exit(1)
    
    project_id = sys.argv[1]
    email = None
    if "--email" in sys.argv:
        idx = sys.argv.index("--email")
        if idx + 1 < len(sys.argv):
            email = sys.argv[idx + 1]
    
    print("=" * 60)
    print("  TWCC Configuration")
    print("  Project ID: %s" % project_id)
    if email:
        print("  Email: %s" % email)
    print("=" * 60)
    
    # Find all .sh files in twcc/
    twcc_dir = os.path.dirname(os.path.abspath(__file__))
    sh_files = glob.glob(os.path.join(twcc_dir, "*.sh"))
    
    for fpath in sh_files:
        with open(fpath, "r") as f:
            content = f.read()
        
        modified = False
        
        # Replace PROJECT_ID
        if "PROJECT_ID" in content:
            content = content.replace(
                "#SBATCH --account=PROJECT_ID",
                "#SBATCH --account=%s" % project_id
            )
            modified = True
        
        # Enable and set email
        if email:
            content = content.replace(
                "# #SBATCH --mail-user=你的email@xxx.com",
                "#SBATCH --mail-user=%s" % email
            )
            modified = True
        
        if modified:
            with open(fpath, "w") as f:
                f.write(content)
            print("  Updated: %s" % os.path.basename(fpath))
    
    print("\nDone! All SLURM scripts configured.")
    print("\nNext steps:")
    print("  1. git add twcc/ && git commit -m 'Configure TWCC project ID'")
    print("  2. git push")
    print("  3. SSH to TWCC and git pull")


if __name__ == "__main__":
    main()
