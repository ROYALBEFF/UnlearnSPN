import numpy as np
import pickle
import abc
from pathlib import Path


class Data:
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def store(arr, path):
        with open(path, 'wb+') as f:
            pickle.dump(arr, f)

    @staticmethod
    @abc.abstractmethod
    def preprocess(path):
        return


class MSNBC(Data):
    @staticmethod
    def preprocess(path):
        with open(path) as f:
            sequences = list(f)[7:]
            data = np.zeros([len(sequences), 17], dtype=np.int)
            for i, s in enumerate(sequences):
                pages = set([int(x) for x in s.split(' ')[:-1]])
                for j in pages:
                    data[i, j - 1] = 1
        return data


class Plants(Data):
    @staticmethod
    def preprocess(path):
        states = ['ab', 'ak', 'ar', 'az', 'ca', 'co', 'ct', 'de', 'dc', 'of', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv',
                  'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'pr', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'vi', 'wa', 'wv', 'wi', 'wy', 'al', 'bc', 'mb', 'nb', 'lb', 'nf',
                  'nt', 'ns', 'nu', 'on', 'pe', 'qc', 'sk', 'yt', 'dengl', 'fraspm']
        state_ids = {st: i for i, st in enumerate(states)}

        with open(path, encoding='ISO-8859-1') as f:
            rows = []
            plant_count = 0
            for line in f:
                line = line.replace('\n', '')
                line = line.replace(',gl', ',dengl')
                values = line.split(',')

                if len(values) == 1:
                    continue

                ids = [state_ids[st] for st in values[1:]]
                row = np.zeros(len(states) + 1)
                row[0] = plant_count
                row[1:][ids] = 1
                rows.append(row)
                plant_count += 1

        data = np.array(rows)
        return data


class Adult(Data):
    @staticmethod
    def preprocess(path):
        with open(path) as f:
            workclasses = '?, Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'
            workclasses = {wc: i for i, wc in enumerate(workclasses.split(', '))}
            educations = '?, Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'
            educations = {ed: i for i, ed in enumerate(educations.split(', '))}
            martial_statuses = '?, Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'
            martial_statuses = {ms: i for i, ms in enumerate(martial_statuses.split(', '))}
            occupations = '?, Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, ' \
                          'Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'
            occupations = {os: i for i, os in enumerate(occupations.split(', '))}
            relationships = '?, Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'
            relationships = {rs: i for i, rs in enumerate(relationships.split(', '))}
            races = '?, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'
            races = {rs: i for i, rs in enumerate(races.split(', '))}
            sexes = '?, Female, Male'
            sexes = {xs: i for i, xs in enumerate(sexes.split(', '))}
            native_countries = '?, United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, ' \
                               'Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, ' \
                               'Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'
            native_countries = {ncs: i for i, ncs in enumerate(native_countries.split(', '))}
            fiftyk = '>50K, <=50K'
            fiftyk = {f: i for i, f in enumerate(fiftyk.split(', '))}

            rows = []
            for line in f:
                line = line.replace('\n', '')
                values = line.split(', ')
                if len(values) == 1:
                    continue
                age, workclass, fnlwgt, education, education_num, martial_status, occupation, relationship, race, sex, captial_gain, captial_loss, hours_per_week, native_country, fk = values
                row = [int(age), workclasses[workclass], int(fnlwgt), educations[education], int(education_num), martial_statuses[martial_status], occupations[occupation],
                       relationships[relationship], races[race], sexes[sex], int(captial_gain), int(captial_loss), int(hours_per_week), native_countries[native_country], fiftyk[fk]]
                rows.append(row)

            data = np.array(rows)
            return data


class Abalone(Data):
    @staticmethod
    def preprocess(path):
        with open(path) as f:
            rows = []
            for line in f:
                line = line.replace('\n', '')
                values = line.split(',')
                values[0] = 0 if values[0] == 'M' else 1 if values[0] == 'F' else 2
                for i in range(1, 9):
                    values[i] = float(values[i])
                values[-1] = int(values[-1])
                rows.append(values)

            data = np.array(rows)
            return data


class Wine(Data):
    @staticmethod
    def preprocess(path):
        with open(path) as f:
            rows = []
            for line in f:
                line = line.replace('\n', '')
                values = line.split(',')
                for i in range(len(values)):
                    values[i] = float(values[i])
                rows.append(values)

            data = np.array(rows)
            return data


class DataLoader:
    def __init__(self):
        from spn.structure.Base import Context
        from spn.structure.leaves.parametric.CRParametric import CRCategorical, CRGaussian
        from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
        self._datasets = {
            'msnbc': (MSNBC, 'data/MSNBC/data.pkl', 'data/MSNBC/msnbc990928.seq'),
            'plants': (Plants, 'data/Plants/data.pkl', 'data/Plants/plants.data'),
            'adult': (Adult, 'data/Adult/data.pkl', 'data/Adult/adult.data'),
            'abalone': (Abalone, 'data/Abalone/data.pkl', 'data/Abalone/abalone.data'),
            'wine': (Wine, 'data/Wine/data.pkl', 'data/Wine/wine.data')
        }

        self._cr_enabled_contexts = {
            'msnbc': Context(parametric_types=[CRCategorical] * 17),
            'plants': Context(parametric_types=[CRCategorical] * 71),
            'adult': Context(parametric_types=[CRGaussian, CRCategorical, CRGaussian, CRCategorical, CRGaussian, CRCategorical, CRCategorical, CRCategorical, CRCategorical, CRCategorical, CRGaussian,
                                               CRGaussian, CRGaussian, CRCategorical, CRCategorical]),
            'abalone': Context(parametric_types=[CRCategorical] + [CRGaussian] * 8),
            'wine': Context(parametric_types=[CRCategorical] + [CRGaussian]*13)
        }

        self._contexts = {
            'msnbc': Context(parametric_types=[Categorical] * 17),
            'plants': Context(parametric_types=[Categorical] * 71),
            'adult': Context(parametric_types=[Gaussian, Categorical, Gaussian, Categorical, Gaussian, Categorical, Categorical, Categorical, Categorical, Categorical, Gaussian, Gaussian, Gaussian,
                                               Categorical, Categorical]),
            'abalone': Context(parametric_types=[Categorical] + [Gaussian] * 8),
            'wine': Context(parametric_types=[Categorical] + [Gaussian] * 13)
        }

    def load(self, dataset, shuffle=False, seed=None):
        dataset, pickle_path, raw_path = self._datasets[dataset.lower()]
        pickle_path = Path(pickle_path)
        if pickle_path.is_file():
            data = Data.load(pickle_path)
        else:
            data = dataset.preprocess(raw_path)
            Data.store(data, pickle_path)

        if shuffle:
            assert seed is not None
            permutation = np.random.RandomState(seed).permutation(data.shape[0])
            data = data[permutation]

        return data

    def context(self, dataset, cr_enabled=True):
        return self._cr_enabled_contexts[dataset.lower()] if cr_enabled else self._contexts[dataset.lower()]
