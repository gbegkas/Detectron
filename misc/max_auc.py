import json
import argparse
import sys
import os
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--path',
        dest='auc_file',
        help='json with auc',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.auc_file) as fp:
        data = json.load(fp)

    max_auc = 0
    for key in sorted(data.keys()):
        if max_auc < data[key]['INbreast']['box']['AUC']:
            max_auc = data[key]['INbreast']['box']['AUC']
            max_key = key
        # if int(key) > 32000:
        #     print(data[key])
    # with open(os.path.join(os.path.dirname(args.auc_file), 'auc.json'), 'w') as fp:
        # json.dump(data, fp, indent=1, separators=(',', ':'), sort_keys=True)

    # print(data[max_key]['INbreast']['box'])

    print('Iteration: {iter}'.format(iter=max_key))
    for metric in data[max_key]['INbreast']['box'].keys():
        print('{metric}: {data}'.format(metric=metric, data=data[max_key]['INbreast']['box'][metric]))

    # print('Auc: {auc}'.format(auc=max_auc))

    keys = map(int, data.keys())
    keys = sorted(keys)
    auc = []
    ap50 = []
    ap = []
    ap75 = []
    apl = []
    apm = []
    aps = []
    for key in keys:
        auc.append(data[str(key)]['INbreast']['box']['AUC'])
        ap50.append(data[str(key)]['INbreast']['box']['AP50'])
        ap.append(data[str(key)]['INbreast']['box']['AP'])
        ap75.append(data[str(key)]['INbreast']['box']['AP75'])
        apl.append(data[str(key)]['INbreast']['box']['APl'])
        apm.append(data[str(key)]['INbreast']['box']['APm'])
        aps.append(data[str(key)]['INbreast']['box']['APs'])

    fig = plt.figure()
    plt.plot(keys, auc)
    # plt.plot(keys, ap, c='brown')
    # plt.plot(keys, ap50, c='black')
    # plt.plot(keys, ap75, c='cyan')
    # plt.plot(keys, apl, c='magenta')
    # plt.plot(keys, apm, c='yellow')
    # plt.plot(keys, aps, c='green')
    # plt.title('AUC Model {model} {dataset}'.format(model=backbone, dataset=cfg.TEST.DATASETS[0]))
    plt.ylabel('Auc')
    plt.xlabel('Iterations')
    plt.axis([min(keys) - 1000, max(keys) + 1000, min(auc) - 0.05, max(auc) + 0.05])
    plt.plot([20000, 50000], [0.94, 0.94], c='red')
    # plt.xlim([min(keys) - 1000, max(keys) + 1000])
    # plt.ylim([min(values) - 0.05, max(values) + 0.05])
    # fig.savefig(os.path.join(cfg.OUTPUT_DIR,
    #                          '{model}_{dataset}_{iter}.eps'.format(model=backbone,
    #                                                                dataset=cfg.TEST.DATASETS[0],
    #                                                                iter=cfg.SOLVER['MAX_ITER'])), format='eps')
    # fig.savefig(os.path.join(cfg.OUTPUT_DIR,
    #                          '{model}_{dataset}_{iter}.png'.format(model=backbone,
    #                                                                dataset=cfg.TEST.DATASETS[0],
    #                                                                iter=cfg.SOLVER['MAX_ITER'])))
    plt.show()
    fig = plt.figure()
    plt.plot(keys, ap50, c='black')
    plt.ylabel('AP50')
    plt.xlabel('Iterations')
    plt.show()
