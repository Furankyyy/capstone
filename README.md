# Capstone Project: News Timeline Summarization

The code for this project is built on the repository created by Ghalandari and Ifrim's work, Examining the State-of-the-Art in News Timeline Summarization (2020), [available here](https://github.com/complementizer/news-tls).


### Dataset
All datasets are [available here](https://drive.google.com/drive/folders/1gDAF5QZyCWnF_hYKbxIzOyjT6MSkbQXu?usp=sharing), including:
* T17
* Crisis
* Entities
Credits to Ghalandari and Ifrim (2020)


### Installation
The `news-tls` library contains tools for loading TLS datasets and running TLS methods.
To install, run:
```
pip install -r requirements.txt
pip install -e .
```
[Tilse](https://github.com/smartschat/tilse) also needs to be installed for evaluation and some TLS-specific data classes.


### Reproduce results

Check reproduce_results.txt for scripts.


### Sample timeline (from DateBART)

For more results comparison, check `result_examples.md`

```
[2010-11-28]
A quarter of a million US state department cables leaked in the Guardian and a number of
---
[2010-11-29]
The Guardian 's editor will be online tomorrow at 4 pm to take questions on the
---
[2010-11-30]
WikiLeaks founder Julian Assange facing growing legal problems around the world .
---
[2010-12-01]
The head of the Bank of England privately criticised David Cameron and George Osborne for their lack
---
[2010-12-02]
New cables released tonight reveal that Washington has called for diplomats in Romania, Hungary and Slovenia
---
[2010-12-03]
French industry minister writes to French body governing internet use warning that there would be consequences for
---
[2010-12-04]
Switzerland rejects growing international calls to force the site off the internet.
---
[2010-12-05]
In China today there is no reference to WikiLeaks material relating to China on news sites .
---
[2010-12-06]
Negotiations on Iran 's nuclear programme restart in Geneva tomorrow in shadow of WikiLeaks
---
[2010-12-07]
WikiLeaks founder Julian Assange remanded in custody today after appearing in court on an extradition warrant
---
[2010-12-08]
MasterCard was partially paralysed today in revenge for the payment network 's decision to
---
[2010-12-09]
PayPal says it is releasing the money held in the WikiLeaks account. Former Croatian prime minister
---
[2010-12-10]
Two leading Pakistani papers admitted today they had been hoaxed by a fake account of WikiLeaks
---
[2010-12-11]
Protests will be held around the world today against the detention of WikiLeaks founder Julian Assange
---
[2010-12-12]
Online shopping site Amazon was briefly offline this evening in the UK, Germany, Italy and
---
[2010-12-13]
MI5 willing to hand over files relating to one of the most high profile murders of
---
[2010-12-14]
WikiLeaks founder Julian Assange refused bail at Westminster magistrates court . He will be remanded
---
[2010-12-15]
Scotland Yard says it has been examining a number of alleged criminal offences by Anonymous .
---
[2010-12-16]
The high court will hold a hearing tomorrow on a Swedish appeal against bail for Julian Assange
---
[2010-12-17]
Tunisia, Saudi Arabia and Morocco have all tried to stem the flow of Wiki - revelations
---
[2010-12-19]
The lack of security at the Yemen facility would be a high priority for the US government
---
[2010-12-20]
BAE system of making secret payments to secure arms contracts will be detailed for the first
---
[2010-12-25]
Batman and Robin scene from Only Fools and Horses voted most memorable Christmas TV moment ever
---
[2011-02-24]
Julian Assange will be extradited to Sweden to answer accusations of rape and sexual assault .
---
[2011-12-16]
Bradley Manning to go before military panel on 16 December at start of most high - profile
---
[2012-06-19]
Assange directed his plea for asylum personally to the Ecuadorian president, Rafael Correa .
---
[2012-08-19]
Assange will give a statement outside the embassy on Sunday afternoon, according to tweets posted on
---
[2013-02-28]
Bradley Manning pleads guilty to 10 charges but still denies aiding the enemy .
---
[2013-06-03]
Bradley Manning is accused of betraying his nation to satisfy craving for notoriety. Army private
---
[2013-06-23]
Snowden is believed to have boarded Aeroflot SU213 on Sunday morning bound for Moscow
---
```