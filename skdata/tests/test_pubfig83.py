import numpy as np

import skdata.pubfig83 as pubfig83


def test_meta():
    dataset = pubfig83.PubFig83()
    meta = dataset.meta
    names = dataset.names.tolist()
    assert names == sorted(names)
    assert len(meta) == 13838
    assert len(names) == 13838
    assert np.unique(names).tolist() == NAMES
    counts = [names.count(n) for n in NAMES]
    assert counts == COUNTS


def test_classification_splits():
    classification_splits_base(nfolds=pubfig83.DEFAULT_NFOLDS,
                               ntrain=pubfig83.DEFAULT_NTRAIN,
                               ntest=pubfig83.DEFAULT_NTEST,
                               nvalidate=pubfig83.DEFAULT_NVALIDATE)

    classification_splits_base(nfolds=2,
                               ntrain=pubfig83.DEFAULT_NTRAIN,
                               ntest=pubfig83.DEFAULT_NTEST,
                               nvalidate=pubfig83.DEFAULT_NVALIDATE)

    classification_splits_base(nfolds=10,
                               ntrain=pubfig83.DEFAULT_NTRAIN,
                               ntest=pubfig83.DEFAULT_NTEST,
                               nvalidate=pubfig83.DEFAULT_NVALIDATE)

    classification_splits_base(nfolds=pubfig83.DEFAULT_NFOLDS,
                               ntrain=40,
                               ntest=pubfig83.DEFAULT_NTEST,
                               nvalidate=pubfig83.DEFAULT_NVALIDATE)

    classification_splits_base(nfolds=pubfig83.DEFAULT_NFOLDS,
                               ntrain=40,
                               ntest=20,
                               nvalidate=pubfig83.DEFAULT_NVALIDATE)

    classification_splits_base(nfolds=pubfig83.DEFAULT_NFOLDS,
                               ntrain=40,
                               ntest=20,
                               nvalidate=0)

    try:
        classification_splits_base(nfolds=pubfig83.DEFAULT_NFOLDS,
                               ntrain=200,
                               ntest=20,
                               nvalidate=pubfig83.DEFAULT_NVALIDATE)
    except pubfig83.NotEnoughExamplesError:
        pass
    else:
        raise Exception('Should have raised exception')


def classification_splits_base(nfolds, ntrain, nvalidate, ntest):
    """
    Test that there are Test and Train/Validate splits
    """
    dataset = pubfig83.PubFig83(ntrain=ntrain, nfolds=nfolds, ntest=ntest,
                                nvalidate=nvalidate)
    splits = dataset.classification_splits
    assert set(splits.keys()) == set(correct_split_names(nfolds))
    assert len(np.unique(splits['Test'])) == 83 * ntest
    names = dataset.names
    assert (names[splits['Test']] == np.repeat(NAMES, ntest)).all()
    for ind in range(nfolds):
        assert len(np.unique(splits['Train%d' % ind])) == ntrain * 83
        assert len(np.unique(splits['Validate%d' % ind])) == nvalidate * 83
        assert (names[splits['Train%d' % ind]]
                             == np.repeat(NAMES, ntrain)).all()
        assert (names[splits['Validate%d' % ind]]
                             == np.repeat(NAMES, nvalidate)).all()
        #no intersections between test & train & validate)
        assert set(splits['Test']).intersection(
                                         splits['Train%d' % ind]) == set([])
        assert set(splits['Test']).intersection(
                                       splits['Validate%d' % ind]) == set([])
        assert set(splits['Train%d' % ind]).intersection(
                                    splits['Validate%d' % ind]) == set([])


def test_classification_task():
    dataset = pubfig83.PubFig83()
    paths, labels, inds = dataset.raw_classification_task()
    assert (np.unique(labels) == range(83)).all()
    assert (labels == np.repeat(range(83), COUNTS)).all()


def test_images():
    dataset = pubfig83.PubFig83()
    I, labels = dataset.img_classification_task()
    #there are 13838 100x100 rgb images
    assert I.shape == (13838, 100, 100, 3)
    #a random sampling of 100 having the right checksums
    rng = np.random.RandomState(0)
    inds = rng.randint(13838, size=(100,))
    assert [I[k].sum() for k in inds] == DATA_SUMS


def correct_split_names(nfolds):
    split_names = ['Test']
    for ind in range(nfolds):
        split_names.append('Train%d' % ind)
        split_names.append('Validate%d' % ind)
    return split_names


DATA_SUMS = [3183920,
 2381895,
 2497078,
 3700911,
 2533567,
 2829736,
 3957948,
 2941275,
 3663414,
 4175522,
 4545230,
 3176163,
 2652150,
 2471430,
 2144770,
 4740294,
 4572341,
 3974616,
 2226700,
 3629392,
 3943478,
 3403567,
 3691577,
 890403,
 4196404,
 3583622,
 5685328,
 3843063,
 4130031,
 3781745,
 3693937,
 4125657,
 1390701,
 1943173,
 3618524,
 4654504,
 4665718,
 3277141,
 3934313,
 3299852,
 4722088,
 3655136,
 4589109,
 3565381,
 4761454,
 4475485,
 3465789,
 3301557,
 5026760,
 3802993,
 2146857,
 2603939,
 4387492,
 1534471,
 2423450,
 2423450,
 3513793,
 2715373,
 3729230,
 2429881,
 3134908,
 3442861,
 2906273,
 3432838,
 2586067,
 3440165,
 2112775,
 2222274,
 3857327,
 3499149,
 3345367,
 2801360,
 3056529,
 4648178,
 5010990,
 3719158,
 2397436,
 4283887,
 5021188,
 3495208,
 2782718,
 3130212,
 3866963,
 4484351,
 3990490,
 4404666,
 4405524,
 4067704,
 2960773,
 2320882,
 3444839,
 3998430,
 4276482,
 3327832,
 4051843,
 4109509,
 4220971,
 3645995,
 3256999,
 3039119]

NAMES = ['Adam Sandler',
 'Alec Baldwin',
 'Angelina Jolie',
 'Anna Kournikova',
 'Ashton Kutcher',
 'Avril Lavigne',
 'Barack Obama',
 'Ben Affleck',
 'Beyonce Knowles',
 'Brad Pitt',
 'Cameron Diaz',
 'Cate Blanchett',
 'Charlize Theron',
 'Christina Ricci',
 'Claudia Schiffer',
 'Clive Owen',
 'Colin Farrell',
 'Colin Powell',
 'Cristiano Ronaldo',
 'Daniel Craig',
 'Daniel Radcliffe',
 'David Beckham',
 'David Duchovny',
 'Denise Richards',
 'Drew Barrymore',
 'Dustin Hoffman',
 'Ehud Olmert',
 'Eva Mendes',
 'Faith Hill',
 'George Clooney',
 'Gordon Brown',
 'Gwyneth Paltrow',
 'Halle Berry',
 'Harrison Ford',
 'Hugh Jackman',
 'Hugh Laurie',
 'Jack Nicholson',
 'Jennifer Aniston',
 'Jennifer Lopez',
 'Jennifer Love Hewitt',
 'Jessica Alba',
 'Jessica Simpson',
 'Joaquin Phoenix',
 'John Travolta',
 'Julia Roberts',
 'Julia Stiles',
 'Kate Moss',
 'Kate Winslet',
 'Katherine Heigl',
 'Keira Knightley',
 'Kiefer Sutherland',
 'Leonardo DiCaprio',
 'Lindsay Lohan',
 'Mariah Carey',
 'Martha Stewart',
 'Matt Damon',
 'Meg Ryan',
 'Meryl Streep',
 'Michael Bloomberg',
 'Mickey Rourke',
 'Miley Cyrus',
 'Morgan Freeman',
 'Nicole Kidman',
 'Nicole Richie',
 'Orlando Bloom',
 'Reese Witherspoon',
 'Renee Zellweger',
 'Ricky Martin',
 'Robert Gates',
 'Sania Mirza',
 'Scarlett Johansson',
 'Shahrukh Khan',
 'Shakira',
 'Sharon Stone',
 'Silvio Berlusconi',
 'Stephen Colbert',
 'Steve Carell',
 'Tom Cruise',
 'Uma Thurman',
 'Victoria Beckham',
 'Viggo Mortensen',
 'Will Smith',
 'Zac Efron']

COUNTS = [108,
 103,
 214,
 171,
 101,
 299,
 268,
 117,
 126,
 300,
 246,
 160,
 195,
 143,
 122,
 134,
 145,
 112,
 168,
 168,
 246,
 187,
 149,
 200,
 152,
 100,
 130,
 135,
 115,
 227,
 102,
 253,
 110,
 150,
 157,
 168,
 101,
 230,
 129,
 107,
 175,
 300,
 108,
 132,
 132,
 132,
 153,
 134,
 257,
 195,
 135,
 199,
 354,
 102,
 108,
 154,
 210,
 146,
 102,
 119,
 367,
 108,
 185,
 188,
 260,
 157,
 133,
 143,
 100,
 128,
 273,
 152,
 201,
 206,
 121,
 124,
 166,
 197,
 167,
 134,
 112,
 128,
 193]
