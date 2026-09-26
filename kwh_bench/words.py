"""Fixed word list for deterministic prompt generation (SPEC.md §4).

Normative for series I-1: do not reorder, add or remove entries. The prompt
set hash pins the output, so any change here fails PROMPT_SET_SHA256.
"""

WORDS = tuple(
    """
the of and to in is that it was for on are as with his they at be this from
have or by one had not but what all were when we there can an your which their
said if do will each about how up out them then she many some so these would
other into has more her two like him see time could no make than first been its
who now people my made over did down only way find use may water long little
very after words called just where most know get through back much before also
around another came come work three word must because does part even place well
such here take why things help put years different away again off went old
number great tell men say small every found still between name should home big
give air line set own under read last never us left end along while might next
sound below saw something thought both few those always looked show large often
together asked house world going want school important until form food keep
children feet land side without boy once animals life enough took sometimes four
head above kind began almost live page got earth need far hand high year mother
light parts country father let night following picture being study second eyes
soon times story boys since white days ever paper hard near sentence better best
across during today others however sure means knew try told young miles sun ways
thing whole hear example heard several change answer room sea against top turned
learn point city play toward five using himself usually money seen car morning
body upon family later turn move face door cut done group true half red fish
plants living black eat short united run book gave order open ground cold really
table remember tree course front american space inside ago sad early ran list
though feel talk bird soon leave family song farm whether money sit horse hours
best states government company system program question during national numbers
process problem business power office public control history report members
available research level community result data model market service policy
project design method value system energy network account material machine
capital region resource industry standard record signal structure product
quality measure balance figure surface pattern element function object volume
section chapter language contract engine motor battery circuit copper carbon
silver river forest valley mountain island harbor bridge station tower garden
market square avenue village castle temple museum library theater kitchen
window mirror curtain blanket candle basket bottle pocket button ribbon needle
thread canvas marble timber gravel cement plaster fabric leather rubber plastic
glass paper cotton wool linen silk velvet amber ivory pearl coral jade opal
ruby onyx quartz granite basalt slate chalk clay sand dust smoke steam frost
thunder lightning breeze drizzle shower rainbow sunrise sunset twilight midnight
harvest orchard meadow pasture prairie desert glacier canyon plateau lagoon
reef delta marsh swamp tundra savanna jungle grove hedge thicket brook creek
pond lake stream waterfall spring fountain well cistern aqueduct canal dam
mill factory foundry workshop studio gallery archive registry ledger journal
notebook manual guide atlas chart diagram sketch draft outline summary review
survey census index catalog inventory schedule agenda budget invoice receipt
voucher permit license charter treaty statute clause verdict appeal motion
petition ballot mandate quorum caucus council senate assembly tribunal jury
witness evidence exhibit testimony alibi motive theory premise axiom lemma
proof corollary theorem formula equation integral vector matrix tensor scalar
gradient derivative limit series sequence interval domain range kernel filter
buffer cache queue stack heap graph tree node edge path cycle loop branch merge
commit release patch version module package library framework runtime compiler
parser lexer token syntax grammar schema query cursor transaction replica shard
cluster gateway proxy router switch adapter socket packet frame header payload
checksum cipher digest salt nonce key lock vault token badge ticket coupon
voucher wallet purse coin bill note bond share stock fund trust estate lease
mortgage tenant landlord broker dealer vendor buyer seller client patron guest
host pilot sailor miner farmer baker weaver potter smith mason carpenter
plumber tailor barber butcher grocer clerk cashier teller auditor actuary
analyst planner scout ranger warden keeper curator editor author poet novelist
critic scholar tutor mentor pupil intern apprentice journeyman master expert
novice amateur veteran rookie captain colonel admiral general marshal sheriff
deputy bailiff usher steward butler valet porter courier envoy herald consul
attache delegate emissary liaison arbiter referee umpire judge coach trainer
athlete runner swimmer climber skater cyclist rower archer fencer wrestler
boxer golfer bowler batter pitcher catcher fielder goalie striker winger
defender midfielder keeper sprinter hurdler jumper vaulter thrower lifter
tumbler dancer singer drummer pianist violinist cellist flutist trumpeter
guitarist bassist conductor composer arranger lyricist producer engineer
technician operator mechanic electrician welder rigger driver conductor
dispatcher navigator surveyor cartographer geologist botanist zoologist
chemist physicist astronomer biologist ecologist economist historian linguist
philosopher theologian architect sculptor painter illustrator animator
photographer filmmaker playwright screenwriter columnist reporter anchor
broadcaster publisher printer binder engraver etcher lithographer typesetter
""".split()
)

assert len(WORDS) >= 500, len(WORDS)
