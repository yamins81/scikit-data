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
    dataset = pubfig83.PubFig83()
    splits = dataset.classification_splits
    assert set(splits.keys()) == set(SPLIT_NAMES)
    assert len(np.unique(splits['Test'])) == 830
    names = dataset.names
    assert (names[splits['Test']] == np.repeat(NAMES, 10)).all()
    for ind in range(5):
        assert len(np.unique(splits['Train%d' % ind])) == 80*83
        assert len(np.unique(splits['Validate%d' % ind])) == 10*83
        assert (names[splits['Train%d' % ind]] == np.repeat(NAMES, 80)).all()
        assert (names[splits['Validate%d' % ind]] == np.repeat(NAMES, 10)).all()
        assert set(splits['Test']).intersection(splits['Train%d' % ind]) == set([])
        assert set(splits['Test']).intersection(splits['Validate%d' % ind]) == set([])
        assert set(splits['Train%d' % ind]).intersection(splits['Validate%d' % ind]) == set([])


def test_classification_task():
    dataset = pubfig83.PubFig83()
    paths, labels, inds = dataset.raw_classification_task()
    assert (np.unique(labels) == range(83)).all()
    assert (labels == np.repeat(range(83), COUNTS)).all()

    
SPLIT_NAMES = ['Test', 'Train0', 'Validate0', 'Train1', 'Validate1', 
               'Train2', 'Validate2', 'Train3', 'Validate3', 'Train4',
               'Validate4']


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