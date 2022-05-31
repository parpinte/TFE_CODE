import traceback
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os, sys

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = event_accumulator.EventAccumulator(path,
            size_guidance={event_accumulator.SCALARS:10000})
        
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

if __name__ == '__main__':

    logdir = 'runs'
    if len(sys.argv) > 1:
        logdir = sys.argv[1]

    for logname in os.listdir(logdir):
        print(logname)
        logfile = os.path.join(logdir,logname)
        logfile = os.path.join(logfile,os.listdir(logfile)[-1])
        df=tflog2pandas(logfile)
        #df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
        df.to_excel(f"data/{logname}.xlsx")
