import pandas as pd
import numpy as np 

class get_task_embeddings():
    def get(self):
        df_exp = pd.read_csv('../../input/gpts_labels/gpt_exposure_embeddings.csv')

        df_exp.task_embedding = df_exp.task_embedding.apply(lambda x: x.strip("[]").split(", "))

        task_embeddings = []
        for x in df_exp.task_embedding:
            task_embeddings.append([float(y) for y in x])
        df_exp.task_embedding = task_embeddings


        task_embeddings = np.array(task_embeddings)
        task_embeddings = np.vstack(task_embeddings)


        df_exp.title_embedding = df_exp.title_embedding.apply(lambda x: x.strip("[]").split(", "))
        title_embeddings = []
        for x in df_exp.title_embedding:
            title_embeddings.append([float(y) for y in x])
        df_exp.title_embedding = title_embeddings
        self.df_exp = df_exp
        return task_embeddings, title_embeddings, df_exp
