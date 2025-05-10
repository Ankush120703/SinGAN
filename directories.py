
import os

def save_dir_tree(start_path='.', indent='', file=None):
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            file.write(f'{indent}📁 {item}/\n')
            save_dir_tree(path, indent + '    ', file)
        else:
            file.write(f'{indent}📄 {item}\n')

with open('project_structure.txt', 'w', encoding='utf-8') as f:
    save_dir_tree('.', '', f)
