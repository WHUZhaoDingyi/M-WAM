"""Run a documented command, retain its exit code, then show results and paths."""
import json
from pathlib import Path
import subprocess
import sys


OUTPUT_FLAGS = {'--output', '--output-dir', '--output-root', '--out', '--visualize-dir'}


def show_result(arguments, returncode):
    print(f'\n[执行结果] {"命令成功" if returncode == 0 else "命令失败"}，退出码={returncode}', flush=True)
    print('后台启动/计划命令的成功不代表后台任务已经完成。', flush=True) if 'start' in arguments or '--dry-run' in arguments else None
    paths = []
    for i, arg in enumerate(arguments):
        flag, sep, value = arg.partition('=')
        if flag in OUTPUT_FLAGS:
            if not sep and i + 1 < len(arguments):
                value = arguments[i + 1]
            if value:
                path = Path(value).expanduser().resolve()
                if path not in paths:
                    paths.append(path)
    if not paths:
        if '--help' in arguments or '-h' in arguments:
            print('[结果] 帮助列表已显示在上方；没有生成结果文件。')
        elif '--check-config' in arguments:
            print('[结果] 配置检查结果见上方；没有启动训练或生成结果文件。')
        else:
            print('[结果] 文本结果见上方；此命令未指定输出文件路径。')
        return
    for path in paths:
        print(f'[输出位置] {path}')
        if not path.exists():
            print('  尚未生成；请查看上方错误或后台任务进度。')
            continue
        summary = path / 'summary.json' if path.is_dir() else path
        if summary.suffix == '.json' and summary.is_file():
            try:
                data = json.loads(summary.read_text())
                if isinstance(data, dict):
                    for key in ('kind', 'method', 'stage', 'status', 'records', 'last_recorded_step',
                                'latest', 'successes', 'completed', 'success_rate', 'exact_identity_cover',
                                'index', 'dataset_size', 'exact_configuration_equal', 'changed_keys',
                                'fixed_input_equivalent', 'training_equivalence'):
                        if key in data:
                            print(f'  {key}: {json.dumps(data[key], ensure_ascii=False)}')
                    if isinstance(data.get('latest_record'), dict):
                        print(f"  当前记录 step: {data['latest_record'].get('step')}")
            except (OSError, ValueError) as error:
                print(f'  摘要无法读取: {error}')
        # Only a bounded list; checkpoints/datasets can be enormous.
        files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
        for file in files[:30]:
            label = {'.png': '图片', '.csv': '表格', '.json': '结果JSON', '.md': '结果说明', '.zip': '压缩包'}.get(file.suffix, '文件')
            print(f'[{label}] {file}')
        if len(files) > 30:
            print(f'  另有 {len(files)-30} 个文件，请查看输出目录。')
        if path.is_dir():
            for child in sorted(p for p in path.iterdir() if p.is_dir()):
                print(f'[子目录] {child}')
        archive = Path(str(path) + '.zip')
        if archive.is_file():
            print(f'[压缩包] {archive}')
    print('以上为输出位置中现有的文件；失败或断点续跑时可能包含先前产物。')


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit('Usage: run_with_result.py -- COMMAND [ARGS...]')
    try:
        code = subprocess.run(arguments, check=False).returncode
    except OSError as error:
        print(f'[启动失败] {error}', file=sys.stderr)
        code = 127
    except KeyboardInterrupt:
        code = 130
    show_result(arguments, code)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
