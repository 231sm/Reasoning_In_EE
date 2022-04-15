import json
import random
from utils.configparser_hook import get_config


def select_word(sample, mention_index):
    left = sample['golden-event-mentions'][0]['trigger']['start']
    right = sample['golden-event-mentions'][0]['trigger']['end']
    cnt = 0
    while True:
        cnt += 1
        if cnt > 100:
            break
        idx = random.randint(0, len(sample['words']) - 1)
        if left <= idx < right:
            continue
        mention_index += 1
        return {"id": f"{index}-n{mention_index}",
                "trigger_word": sample['words'][idx],
                "sent_id": 0, "offset": [idx, idx + 1]}
    return None


if __name__ == '__main__':
    random.seed(42)
    config = get_config('dmcnn.config')
    label2idx = json.load(open('data/labels.json', 'r'))
    num_labels = len(label2idx)
    # Stanford format data
    src_files = ['data/stanford/train_form.json',
                 'data/stanford/test_form.json', 'data/stanford/test_form.json']
    # MAVEN format data
    dst_files = ['./raw/train.jsonl',
                 './raw/valid.jsonl', './raw/test.jsonl']
    index = 1
    for src, dst in zip(src_files, dst_files):
        processed = []
        samples = json.load(open(src, 'r'))
        for sample in samples:
            tmp = {"id": str(index),
                   "title": str(index),
                   "content": [{"sentence": sample['sentence'], "tokens": sample['words']}]}
            mention_index = 1
            if 'test' not in dst:
                tmp["events"] = [{"id": str(index),
                                  "type": sample['golden-event-mentions'][0]['event_type'],
                                  "type_id": label2idx[sample['golden-event-mentions'][0]['event_type']],
                                  "mention":[{"id": f"{index}-p{mention_index}",
                                              "trigger_word": sample['golden-event-mentions'][0]['trigger']['text'],
                                              "sent_id": 0,
                                              "offset": [sample['golden-event-mentions'][0]['trigger']['start'],
                                                         sample['golden-event-mentions'][0]['trigger']['end']]}]
                                  }]
                tmp["negative_triggers"] = []
                k = config.getfloat("train", "ratio") if "train" in dst else config.getfloat(
                    "test", "ratio")
                if k >= 1:
                    for i in range(k):
                        mention_index += 1
                        word = select_word(sample, mention_index)
                        if not word:
                            break
                        tmp['negative_triggers'].append(word)
                else:
                    if random.random() < k:
                        mention_index += 1
                        word = select_word(sample, mention_index)
                        if word:
                            tmp['negative_triggers'].append(word)
            else:
                tmp["candidates"] = [{"id": f"{index}-c{mention_index}",
                                      "trigger_word": sample['golden-event-mentions'][0]['trigger']['text'],
                                      "sent_id": 0,
                                      "offset": [sample['golden-event-mentions'][0]['trigger']['start'],
                                                 sample['golden-event-mentions'][0]['trigger']['end']]
                                      }]

            index += 1
            processed.append(tmp)

        # Output to dst file
        with open(dst, 'w') as f:
            for x in processed:
                f.write(json.dumps(x) + '\n')
