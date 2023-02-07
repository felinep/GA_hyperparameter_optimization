import pickle, string
from pymoo.visualization.pcp import PCP
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

def store_video(video_name : string, pickle_name : string):
    with Recorder(Video(video_name)) as rec:
        history_list = pickle.load( open(pickle_name, "rb"))

        # for each algorithm object in the history
        generation = 1
        for result_history in history_list:
            for entry in result_history:
                pcp = PCP(title=(f'Generation {generation}, Best validation accuracy: {(- entry.opt.get("F")[0][0]):0.03f}', {'pad': 30},),
                        #legend=(True, {'loc': "upper left"}),
                        bounds=(entry.problem.xl, entry.problem.xu),
                        labels=[f"$layer_{k + 1}$" for k in range(entry.problem.n_var)]
                        )
                generation = generation + 1
                pcp.set_axis_style(color="grey", alpha=0.5)
                
                # get parameters of each population per generation
                pcp.add(entry.pop.get("X"), color="black", alpha=0.8, linewidth=1)
                # print the offsprings blue!
                if entry.off is not None:
                    pcp.add(entry.off.get("X"), color="deepskyblue", alpha=0.8, linewidth=0.5)
                # print the best entry red
                pcp.add(entry.opt.get("X"), color="red", linewidth=1)

                pcp.do()

                rec.record()