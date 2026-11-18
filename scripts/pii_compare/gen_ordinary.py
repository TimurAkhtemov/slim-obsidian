"""Build data/ordinary.json: ordinary sentences with numbers (prices, dates, phones, order numbers,
zips, counts, lists, versions, cue words used for something else), as a digit-writing transcriber
renders them. Generators from the detector's false-positive hunts; emails and websites left out.
Frozen by seed: run once, commit the JSON."""
import json
import random
import sys

R = random.Random(20261118)
ri = R.randint
ch = R.choice

def turn(s):
    return f"**{ch(['Me','Them'])}:** {s}"

def dg(n, lead="123456789"):
    return ch(lead) + "".join(ch("0123456789") for _ in range(n - 1))

F = {}

def fam(name):
    def deco(f):
        F[name] = f
        return f
    return deco

@fam("prices")
def _():
    return ch([
        f"It came to ${ri(1,9)},{ri(100,999)}.{ri(10,99)} with tax.",
        f"They're {ri(2,6)} for {ri(1,49)}.{ri(10,99)} at the store.",
        f"The total was {ri(100,9999)}.{ri(10,99)}, then {ri(10,99)} off.",
        f"I paid {ri(10,999)} dollars and {ri(10,99)} cents.",
        f"It was {ri(1000,99999)} for the car and {ri(100,999)} for the insurance.",
        f"Rent is {ri(1200,4800)}, utilities {ri(80,400)}, internet {ri(30,120)}.",
        f"We spent {ri(1,9)},{ri(100,999)} on flights and {ri(100,999)} on hotels.",
    ])

@fam("dates_times")
def _():
    y = ri(1990, 2030)
    return ch([
        f"The meeting is on {ri(1,12)}/{ri(1,28)}/{y} at {ri(1,12)}:{ri(10,59)}.",
        f"I worked there from {y} to {y+ri(1,8)}.",
        f"See you at {ri(1,12)}:{ri(10,59)}, or {ri(1,12)}:{ri(10,59)} at the latest.",
        f"It was due {y}-{ri(1,12):02d}-{ri(1,28):02d}, then pushed.",
        f"Born {ri(1,12)} {ri(1,28)} {y}.",
        f"Between {ri(1,12)} {ri(10,59)} and {ri(1,12)} {ri(10,59)} works for me.",
        f"On {ri(1,12)}/{ri(1,28)} we shipped {ri(10,999)} units.",
        f"From {y} through {y+1} and {y+2} we grew.",
        f"The {ri(1,28)}th of {ch(['March','May','June'])} {y} at {ri(1,12)} {ri(10,59)}.",
    ])

def phone():
    a, e, n = dg(3, "23456789"), dg(3, "23456789"), f"{ri(0,9999):04d}"
    return ch([f"{a} {e} {n}", f"{a}-{e}-{n}", f"({a}) {e}-{n}", f"{a}.{e}.{n}", f"{a}{e}{n}",
               f"+1 {a} {e} {n}", f"1-{a}-{e}-{n}", f"{e}-{n}", f"{a} {e}{n}", f"{a}, {e}, {n}",
               f"1 {a} {e} {n}", f"{e} {n}"])

@fam("phones_off")
def _():
    return turn(ch([
        f"Call me at {phone()}.", f"You can reach the office on {phone()}.",
        f"My number is {phone()}.", f"{phone()}. That's the front desk.",
        f"Try {phone()} or {phone()}.", f"The pharmacy line is {phone()}, extension {ri(10,999)}.",
    ]))

def order():
    n = ri(6, 20)
    s = dg(n)
    shape = ch(["contig", "4s", "3s", "dash", "amz"])
    if shape == "4s":
        s = " ".join(s[i:i+4] for i in range(0, n, 4))
    elif shape == "3s":
        s = " ".join(s[i:i+3] for i in range(0, n, 3))
    elif shape == "dash":
        s = "-".join(s[i:i+4] for i in range(0, n, 4))
    elif shape == "amz":
        s = f"{dg(3)}-{dg(7)}-{dg(7)}"
    return s

@fam("order_numbers")
def _():
    return turn(ch([
        f"The order number is {order()}.", f"Tracking is {order()}.", f"Ticket {order()}, can you check it?",
        f"Invoice {order()} was paid.", f"Your confirmation number is {order()}.", f"It's {order()}.",
        f"Here it is. {order()}.", f"Reference {order()} on the receipt.",
    ]))

@fam("zip")
def _():
    z = f"{ri(0,99999):05d}"
    return ch([f"The ZIP is {z}.", f"It's {z}-{ri(0,9999):04d}.", f"Ship to {z}{ri(0,9999):04d}.",
               f"ZIP {z} {ri(0,9999):04d}.", f"Zip code {z}, not {ri(10000,99999)}."])

@fam("measurements")
def _():
    return ch([f"It's {ri(1,300)} by {ri(1,300)} by {ri(1,300)} centimeters.",
               f"I ran {ri(1,42)}.{ri(1,9)} kilometers in {ri(20,300)} minutes.",
               f"It weighs {ri(1,999)} grams, {ri(1,99)} ounces.",
               f"The room is {ri(100,999)} square feet, {ri(8,14)} foot ceilings.",
               f"Set it to {ri(100,500)} degrees for {ri(10,90)} minutes."])

@fam("percentages")
def _():
    return ch([f"Up {ri(1,99)}% from {ri(1,99)}%.", f"It's {ri(1,99)}.{ri(1,9)} percent, {ri(1,99)} percent last year.",
               f"We hit {ri(10,99)}%, {ri(10,99)}%, {ri(10,99)}% and {ri(10,99)}%."])

@fam("votes_scores")
def _():
    return ch([f"The vote was {ri(10,999)} to {ri(10,999)}.", f"Final score {ri(0,150)} {ri(0,150)}.",
               f"We got {ri(100,9999)} votes, they got {ri(100,9999)}.",
               f"The counts were {ri(100,999)} {ri(10,99)} {ri(1000,9999)}.",
               f"Results: {ri(100,9999)}, {ri(100,9999)}, {ri(100,9999)}."])

def numlist(sep, k=None):
    k = k or ri(2, 8)
    return sep.join(str(ri(0, ch([9, 99, 999, 9999]))) for _ in range(k))

@fam("number_lists_space")
def _():
    return ch([f"The counts were {numlist(' ')}.", f"Okay. {numlist(' ')}.", f"We had {numlist(' ')} people."])

@fam("number_lists_comma")
def _():
    return ch([f"The counts were {numlist(', ')}.", f"Okay. {numlist(', ')}.", f"We had {numlist(', ')} people."])

@fam("ip_versions")
def _():
    return ch([f"The server is {ri(1,255)}.{ri(0,255)}.{ri(0,255)}.{ri(0,255)}.",
               f"Upgrade to {ri(1,20)}.{ri(0,20)}.{ri(0,99)}.", f"Version {ri(1,9)} {ri(0,20)} {ri(0,99)} is out.",
               f"Build {ri(1000,99999)}.{ri(0,999)}."])

@fam("flights_rooms_ext")
def _():
    return ch([f"Flight {ch(['UA','DL','AA'])} {ri(10,9999)} leaves from gate {ch('ABCD')}{ri(1,40)}.",
               f"I'm in room {ri(100,999)}, extension {ri(1000,9999)}.",
               f"Flight {ri(100,9999)}, seat {ri(1,40)}{ch('ABCDEF')}, room {ri(100,999)}.",
               f"Bus {ri(1,99)} then {ri(100,999)}, platform {ri(1,12)}."])

@fam("triples_324_333")
def _():
    return turn(ch([f"The rooms were {ri(100,999)} {ri(10,99)} {ri(1000,9999)}.",
                    f"Okay. {ri(100,999)} {ri(10,99)} {ri(1000,9999)}.",
                    f"Scores {ri(100,999)} {ri(100,999)} {ri(100,999)}.",
                    f"Here it is. {ri(100,999)} {ri(100,999)} {ri(100,999)}.",
                    f"Order {dg(9)}, thanks.", f"It's {dg(9,'0123456789')}."]))

@fam("cue_social")
def _():
    return turn(ch([
        f"The Social Security office is open 9 to 5, call {phone()}.",
        f"Call the social security office at {phone()}.",
        f"Our social media post got {ri(100000,99999999)} views.",
        f"The social had {ri(100,999)}, {ri(10,99)}, {ri(1000,9999)} people over three nights.",
        f"Social club dues went from {ri(100,999)} to {ri(100,999)} in {ri(2000,2026)}.",
        f"My social security check came on {ri(2020,2026)}-{ri(1,12):02d}-{ri(1,28):02d}.",
        f"Social media followers: {numlist(', ')}.",
        f"The social worker visited {ri(1,12)}/{ri(1,28)}, order {dg(ri(7,11))}.",
    ]))

@fam("cue_card")
def _():
    return turn(ch([
        f"My credit score is {ri(300,850)}.",
        f"I put the card on the table at {ri(1000,9999)} Main Street.",
        f"Card game at 7, 8 or 9.",
        f"I paid by card, order {order()}.",
        f"The gift card order number is {dg(3)}-{dg(7)}-{dg(7)}.",
        f"Credit hours were {', '.join(str(ri(1,4)) for _ in range(ri(6,14)))}.",
        f"We dealt {numlist(', ')} cards.",
        f"Credit union members: {ri(1000,9999)}, {ri(1000,9999)}, {ri(1000,9999)}, {ri(1000,9999)}.",
        f"The debit went through, confirmation {order()}.",
        f"Card {ri(1,52)} of 52, {ri(2019,2026)} {ri(1000,9999)} {ri(1000,9999)}.",
        f"Extra credit: {numlist(' ')} points.",
        f"Visa appointment on {ri(1,12)}/{ri(1,28)}/{ri(2025,2027)}, case {dg(ri(12,16))}.",
    ]))

@fam("cue_account")
def _():
    return turn(ch([
        f"Account for {ri(1,99)} of the {ri(100000,9999999)} errors.",
        f"The routing table has {ri(100,999)} entries.",
        f"The routing table has {ri(100,999)} entries and {ri(100000,9999999)} lookups.",
        f"Your account number is fine, the order was {order()}.",
        f"The account number changed on {ri(2020,2026)}-{ri(1,12):02d}-{ri(1,28):02d}.",
        f"Call {phone()} about the account number.",
        f"Routing was {numlist(', ')} packets per second.",
        f"Account numbers {numlist(', ')} were flagged.",
        f"Sort code questions? Ticket INC{dg(ri(6,9))}.",
    ]))

@fam("password_cue_words")
def _():
    return turn(ch([
        f"Pin the tab with the {ri(2019,2026)} numbers.",
        f"The password policy changed in {ri(2019,2026)}, now it's {ri(8,16)} characters.",
        f"What's the passcode situation for the {ri(2,9)} new hires?",
        f"Bowling pins cost {ri(10,99)} dollars each, we bought {ri(2,9)}.",
        f"The door code thing is broken again, Dave said call facilities.",
        f"Your login is fine, the server was down from {ri(1,12)} to {ri(1,12)}.",
    ])) + "\n\n" + turn(ch(["Okay, sounds good.", f"Sure, room {ri(100,999)} it is.", "Great, thanks Dave."]))


T2 = {
 # social
 "ss office + phone": lambda: f"The Social Security office is open 9 to 5, call {phone()}.",
 "social media views (contig)": lambda: f"Our social media post got {ri(1000000,99999999)} views.",
 "social + comma triple": lambda: f"The social had {ri(100,999)}, {ri(10,99)}, {ri(1000,9999)} people over three nights.",
 "social + comma list": lambda: f"Social media followers: {numlist(', ', ri(3,6))}.",
 "social + ISO date": lambda: f"My social security check came on {ri(2020,2026)}-{ri(1,12):02d}-{ri(1,28):02d}.",
 "social + order 7-11": lambda: f"The social worker visited on Monday, order {dg(ri(7,11))}.",
 "phone, 'social' 50 chars before": lambda: f"I left social stuff aside for now. Anyway, call {phone()}.",
 # card
 "credit score": lambda: f"My credit score is {ri(300,850)}.",
 "card game 7,8,9": lambda: "Card game at 7, 8 or 9.",
 "card + amazon order 3-7-7": lambda: f"I paid by card, order {dg(3)}-{dg(7)}-{dg(7)}.",
 "card + order 12-19": lambda: f"I paid by card, the order number is {dg(ri(12,19))}.",
 "credit hours 1-digit list": lambda: f"Credit hours were {', '.join(str(ri(1,4)) for _ in range(ri(6,14)))}.",
 "cards dealt list": lambda: f"We dealt {numlist(', ')} cards.",
 "credit union 4x4-digit": lambda: f"Credit union members: {ri(1000,9999)}, {ri(1000,9999)}, {ri(1000,9999)}, {ri(1000,9999)}.",
 "debit + confirmation": lambda: f"The debit went through, confirmation {order()}.",
 "visa appt + case 12-16": lambda: f"Visa appointment on {ri(1,12)}/{ri(1,28)}/{ri(2025,2027)}, case {dg(ri(12,16))}.",
 "credit + two phones": lambda: f"The credit desk is {ri(200,999)} {ri(200,999)} {ri(1000,9999)} or {ri(200,999)} {ri(200,999)} {ri(1000,9999)}.",
 # account
 "account for N of M errors": lambda: f"Account for {ri(1,99)} of the {ri(100000,9999999)} errors.",
 "routing table entries": lambda: f"The routing table has {ri(100,999)} entries.",
 "routing + lookups": lambda: f"The routing table has {ri(100,999)} entries and {ri(100000,9999999)} lookups.",
 "account number + order": lambda: f"Your account number is fine, the order was {order()}.",
 "account number + ISO date": lambda: f"The account number changed on {ri(2020,2026)}-{ri(1,12):02d}-{ri(1,28):02d}.",
 "phone + account number": lambda: f"Call {phone()} about the account number.",
 "routing + comma list": lambda: f"Routing was {numlist(', ', ri(3,6))} packets per second.",
 "sort code + ticket": lambda: f"Sort code questions? Ticket INC{dg(ri(6,9))}.",
 # 324 / 333 no cue
 "counts 3-2-4 spaced (no cue)": lambda: f"The counts were {ri(100,999)} {ri(10,99)} {ri(1000,9999)}.",
 "scores 3-3-3 spaced (no cue)": lambda: f"Scores {ri(100,999)} {ri(100,999)} {ri(100,999)}.",
 "9-digit order contig (no cue)": lambda: f"Order {dg(9)}, thanks.",
 "9-digit leading 0 (no cue)": lambda: f"It's 0{dg(8,'0123456789')}.",
 "ZIP+4 glued (no cue)": lambda: f"Ship to {ri(0,99999):05d}{ri(0,9999):04d}.",
 "16-digit tracking (no cue)": lambda: f"Tracking is {dg(16)}.",
 "16-digit 4-4-4-4 ref (no cue)": lambda: (lambda s: f"Reference {s[:4]} {s[4:8]} {s[8:12]} {s[12:]}.")(dg(16)),
 "4x4-digit comma list starting 4 (no cue)": lambda: f"The counts were {ri(4000,4999)}, {ri(1000,9999)}, {ri(1000,9999)}, {ri(1000,9999)}.",
 "space list 13-19 digits total (no cue)": lambda: f"Okay. {ri(1000,9999)} {ri(100,999)} {ri(1000,9999)} {ri(10,99)} {ri(1000,9999)}.",
}


def site():
    return ch(["amazon.com", "Target.com", "bestbuy.com", "github.io", "bbc.co.uk", "apple.com"])


def main(out):
    rows = []
    for name, gen in F.items():
        for k in range(10):
            rows.append((f"fp:{name}", gen()))
    for name, gen in T2.items():
        for k in range(4):
            rows.append((f"fp2:{name}", gen()))
    cases = []
    for n, (family, text) in enumerate(rows):
        if not text.startswith("**"):
            text = "**Them:** " + text
        cases.append({"id": f"o{n:03d}", "family": family, "split": "test", "text": text})
    json.dump(cases, open(out, "w"), indent=1, ensure_ascii=False)
    print(len(cases), "sentences")


if __name__ == "__main__":
    main(sys.argv[1])
