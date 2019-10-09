#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate the postgres component
"""


def test():
    # access the postgres package
    import bizbook

    # build a database component
    db = bizbook.sqlite()

    # build some locations
    location = bizbook.schema.Location
    locations = (
        location.pyre_immutable(
            id="p00", address="1 First Avenue", city="Boston", state="MA"),
        location.pyre_immutable(
            id="p01", address="2 Second Avenue", city="Washington", state="DC"),
        location.pyre_immutable(
            id="p02", address="3 Third Avenue", city="Berkeley", state="CA"),

        location.pyre_immutable(
            id="a00", address="10932 Bigge Rd.", city="Menlo Park", state="CA", zip="94025"),
        location.pyre_immutable(
            id="a01", address="309 63rd St.", city="Oakland", state="CA", zip="94618"),
        location.pyre_immutable(
            id="a02", address="589 Darwin Ln.", city="Berkeley", state="CA", zip="94705"),
        location.pyre_immutable(
            id="a03", address="22 Cleveland Av.", city="San Jose", state="CA", zip="95128"),
        location.pyre_immutable(
            id="a04", address="5420 College Av.", city="Oakland", state="CA", zip="94609"),
        location.pyre_immutable(
            id="a05", address="10 Mississippi Dr.", city="Lawrence", state="KS", zip="6044"),
        location.pyre_immutable(
            id="a06", address="6223 Bateman St.", city="Berkeley", state="CA", zip="94705"),
        location.pyre_immutable(
            id="a07", address="3410 Blonde St.", city="Palo ALto", state="CA", zip="94301"),
        location.pyre_immutable(
            id="a08", address="PO Box 792", city="Covelo", state="CA", zip="95428"),
        location.pyre_immutable(
            id="a09", address="18 Broadway Av.", city="San Francisco", state="CA", zip="94130"),
        location.pyre_immutable(
            id="a10", address="22 Graybar Rd.", city="Nashville", state="TN", zip="37215"),
        location.pyre_immutable(
            id="a11", address="55 Hillsdale Bl.", city="Corvalis", state="OR", zip="97330"),
        location.pyre_immutable(
            id="a12", address="3 Silver Ct.", city="Walnut Creek", state="CA", zip="94595"),
        location.pyre_immutable(
            id="a13", address="2286 Cram Pl.", city="Ann Arbor", state="MI", zip="48105"),
        location.pyre_immutable(
            id="a14", address="3 Balding Pl.", city="Gary", state="IN", zip="46403"),
        location.pyre_immutable(
            id="a15", address="5420 Telegraph Av.", city="Oakland", state="CA", zip="94609"),
        location.pyre_immutable(
            id="a16", address="44 Upland Hts.", city="Oakland", state="CA", zip="94612"),
        location.pyre_immutable(
            id="a17", address="5720 McAuley St.", city="Oakland", state="CA", zip="94609"),
        location.pyre_immutable(
            id="a18", address="1956 Arlington Pl.", city="Rockville", state="MD", zip="20853"),
        location.pyre_immutable(
            id="a19", address="301 Putnam", city="Vacaville", state="CA", zip="95688"),
        location.pyre_immutable(
            id="a20", address="67 Seventh Ave.", city="Salt Lake City", state="UT", zip="84152"),

        location.pyre_immutable(
            id="e00", address="18 Dowdy Ln.", city="Boston", state="MA", zip="02210"),
        location.pyre_immutable(
            id="e01", address="3000 6th Str.", city="Berkeley", state="CA", zip="94710"),
        location.pyre_immutable(
            id="e02", address="15 Sail", city="Denver", state="CO", zip="80237"),
        location.pyre_immutable(
            id="e03", address="27 Yosemite", city="Oakland", state="CA", zip="94609"),
        location.pyre_immutable(
            id="e04", address="1010 E. Devon", city="Chicago", state="IL", zip="60018"),
        location.pyre_immutable(
            id="e05", address="97 Bleaker", city="Boston", state="MA", zip="02210"),
        location.pyre_immutable(
            id="e06", address="32 Rockbill Pike", city="Rockbill", state="MD", zip="20852"),
        location.pyre_immutable(
            id="e07", address="18 Severe Rd.", city="Berkeley", state="CA", zip="94710"),
        )
    # add them to their table
    for location in locations: db.insert(location)

    # build the publishers
    publisher = bizbook.schema.Publisher
    publishers = (
        publisher.pyre_immutable(id="0736", name="New Age Books", headquarters="p00"),
        publisher.pyre_immutable(id="0877", name="Binnet & Hardley", headquarters="p01"),
        publisher.pyre_immutable(id="1389", name="Algodata Infosystems", headquarters="p02"),
        )
    # add them to their table
    for publisher in publishers: db.insert(publisher)

    # build the persons
    person = bizbook.schema.Person
    people = (
        person.pyre_immutable(ssn="173-32-1176", lastname="White", firstname="Johnson"),
        person.pyre_immutable(ssn="213-46-8915", lastname="Green", firstname="Marjorie"),
        person.pyre_immutable(ssn="238-95-7766", lastname="Carson", firstname="Cheryl"),
        person.pyre_immutable(ssn="267-41-2394", lastname="O'Leary", firstname="Michael"),
        person.pyre_immutable(ssn="274-80-9391", lastname="Straight", firstname="Dick"),
        person.pyre_immutable(ssn="341-22-1782", lastname="Smith", firstname="Meander"),
        person.pyre_immutable(ssn="409-56-7008", lastname="Bennet", firstname="Abraham"),
        person.pyre_immutable(ssn="427-17-2319", lastname="Dull", firstname="Ann"),
        person.pyre_immutable(ssn="472-27-2349", lastname="Gringlesby", firstname="Burt"),
        person.pyre_immutable(ssn="486-29-1786", lastname="Locksley", firstname="Chastity"),
        person.pyre_immutable(ssn="527-72-3246", lastname="Greene", firstname="Morningstar"),
        person.pyre_immutable(ssn="648-92-1872", lastname="Blotchet-Halls", firstname="Reginald"),
        person.pyre_immutable(ssn="672-71-3249", lastname="Yokomoto", firstname="Akiko"),
        person.pyre_immutable(ssn="712-45-1867", lastname="del Castillo", firstname="Innes"),
        person.pyre_immutable(ssn="722-51-5454", lastname="DeFrance", firstname="Michel"),
        person.pyre_immutable(ssn="724-08-9931", lastname="Stringer", firstname="Dirk"),
        person.pyre_immutable(ssn="724-80-9391", lastname="MacFeather", firstname="Stearns"),
        person.pyre_immutable(ssn="756-30-7391", lastname="Karsen", firstname="Livia"),
        person.pyre_immutable(ssn="807-91-6654", lastname="Panteley", firstname="Sylvia"),
        person.pyre_immutable(ssn="846-92-7186", lastname="Hunter", firstname="Sheryl"),
        person.pyre_immutable(ssn="893-72-1158", lastname="McBadden", firstname="Heather"),
        person.pyre_immutable(ssn="899-46-2035", lastname="Ringer", firstname="Anne"),
        person.pyre_immutable(ssn="998-72-3567", lastname="Ringer", firstname="Albert"),

        person.pyre_immutable(ssn="234-88-9720", lastname="Hunter", firstname="Amanda"),
        person.pyre_immutable(ssn="321-55-8906", lastname="DeLongue", firstname="Martinella"),
        person.pyre_immutable(ssn="723-48-9010", lastname="Sparks", firstname="Manfred"),
        person.pyre_immutable(ssn="777-02-9831", lastname="Samuelson", firstname="Bernard"),
        person.pyre_immutable(ssn="777-66-9902", lastname="Almond", firstname="Alfred"),
        person.pyre_immutable(ssn="826-11-9034", lastname="Himmel", firstname="Eleanore"),
        person.pyre_immutable(ssn="885-23-9140", lastname="Rutherford-Hayes", firstname="Hannah"),
        person.pyre_immutable(ssn="943-88-7920", lastname="Kaspschek", firstname="Chistof"),
        person.pyre_immutable(ssn="993-86-0420", lastname="McCann", firstname="Dennis"),
        )
    # add them to their table
    for person in people: db.insert(person)

    # build the addresses
    address = bizbook.schema.Address
    addresses = (
        address.pyre_immutable(person="173-32-1176", address="a00"),
        address.pyre_immutable(person="213-46-8915", address="a01"),
        address.pyre_immutable(person="238-95-7766", address="a02"),
        address.pyre_immutable(person="267-41-2394", address="a03"),
        address.pyre_immutable(person="274-80-9391", address="a04"),
        address.pyre_immutable(person="341-22-1782", address="a05"),
        address.pyre_immutable(person="409-56-7008", address="a06"),
        address.pyre_immutable(person="427-17-2319", address="a07"),
        address.pyre_immutable(person="472-27-2349", address="a08"),
        address.pyre_immutable(person="486-29-1786", address="a09"),
        address.pyre_immutable(person="527-72-3246", address="a10"),
        address.pyre_immutable(person="648-92-1872", address="a11"),
        address.pyre_immutable(person="672-71-3249", address="a12"),
        address.pyre_immutable(person="712-45-1867", address="a13"),
        address.pyre_immutable(person="722-51-5454", address="a14"),
        address.pyre_immutable(person="724-08-9931", address="a15"),
        address.pyre_immutable(person="724-80-9391", address="a16"),
        address.pyre_immutable(person="756-30-7391", address="a17"),
        address.pyre_immutable(person="807-91-6654", address="a18"),
        address.pyre_immutable(person="846-92-7186", address="a07"),
        address.pyre_immutable(person="893-72-1158", address="a19"),
        address.pyre_immutable(person="899-46-2035", address="a20"),
        address.pyre_immutable(person="998-72-3567", address="a20"),

        address.pyre_immutable(person="234-88-9720", address="e00"),
        address.pyre_immutable(person="321-55-8906", address="e01"),
        address.pyre_immutable(person="723-48-9010", address="e02"),
        address.pyre_immutable(person="777-02-9831", address="e03"),
        address.pyre_immutable(person="777-66-9902", address="e04"),
        address.pyre_immutable(person="826-11-9034", address="e05"),
        address.pyre_immutable(person="885-23-9140", address="e06"),
        address.pyre_immutable(person="943-88-7920", address="e07"),
        address.pyre_immutable(person="993-86-0420", address="e06"),
        )
    # add them to their table
    for address in addresses: db.insert(address)

    contact = bizbook.schema.ContactMethod
    contacts = (
        contact.pyre_immutable(uid="408.496.7223", person="173-32-1176", method="phone"),
        contact.pyre_immutable(uid="415.986.7020", person="213-46-8915", method="phone"),
        contact.pyre_immutable(uid="415.548.7723", person="238-95-7766", method="phone"),
        contact.pyre_immutable(uid="408.286.2428", person="267-41-2394", method="phone"),
        contact.pyre_immutable(uid="415.834.2919", person="274-80-9391", method="phone"),
        contact.pyre_immutable(uid="913.843.0462", person="341-22-1782", method="phone"),
        contact.pyre_immutable(uid="415.658.9932", person="409-56-7008", method="phone"),
        contact.pyre_immutable(uid="415.836.7128", person="427-17-2319", method="phone"),
        contact.pyre_immutable(uid="707.938.6445", person="472-27-2349", method="phone"),
        contact.pyre_immutable(uid="415.585.4620", person="486-29-1786", method="phone"),
        contact.pyre_immutable(uid="615.297.2723", person="527-72-3246", method="phone"),
        contact.pyre_immutable(uid="503.745.6402", person="648-92-1872", method="phone"),
        contact.pyre_immutable(uid="415.935.4228", person="672-71-3249", method="phone"),
        contact.pyre_immutable(uid="615.996.8275", person="712-45-1867", method="phone"),
        contact.pyre_immutable(uid="219.547.9982", person="722-51-5454", method="phone"),
        contact.pyre_immutable(uid="415.843.2991", person="724-08-9931", method="phone"),
        contact.pyre_immutable(uid="415.354.7128", person="724-80-9391", method="phone"),
        contact.pyre_immutable(uid="415.534.9219", person="756-30-7391", method="phone"),
        contact.pyre_immutable(uid="301.946.8853", person="807-91-6654", method="phone"),
        contact.pyre_immutable(uid="415.836.7128", person="846-92-7186", method="phone"),
        contact.pyre_immutable(uid="707.448.4982", person="893-72-1158", method="phone"),
        contact.pyre_immutable(uid="801.826.0752", person="899-46-2035", method="phone"),
        contact.pyre_immutable(uid="801.862.0752", person="998-72-3567", method="phone"),

        contact.pyre_immutable(uid="617.432.5586", person="234-88-9720", method="phone"),
        contact.pyre_immutable(uid="415.843.2222", person="321-55-8906", method="phone"),
        contact.pyre_immutable(uid="303.721.3388", person="723-48-9010", method="phone"),
        contact.pyre_immutable(uid="415.843.6990", person="777-02-9831", method="phone"),
        contact.pyre_immutable(uid="312.699.4177", person="777-66-9902", method="phone"),
        contact.pyre_immutable(uid="617.423.0552", person="826-11-9034", method="phone"),
        contact.pyre_immutable(uid="301.468.3909", person="885-23-9140", method="phone"),
        contact.pyre_immutable(uid="415.549.3909", person="943-88-7920", method="phone"),
        contact.pyre_immutable(uid="301.468.3909", person="993-86-0420", method="phone"),
        )
    # add them to their table
    for contact in contacts: db.insert(contact)

    staff = bizbook.schema.Staff
    people = (
        staff.pyre_immutable(person="234-88-9720", position="acquisition"),
        staff.pyre_immutable(person="321-55-8906", position="project"),
        staff.pyre_immutable(person="723-48-9010", position="copy"),
        staff.pyre_immutable(person="777-02-9831", position="project"),
        staff.pyre_immutable(person="777-66-9902", position="copy"),
        staff.pyre_immutable(person="826-11-9034", position="project"),
        staff.pyre_immutable(person="885-23-9140", position="project"),
        staff.pyre_immutable(person="943-88-7920", position="acquisition"),
        staff.pyre_immutable(person="993-86-0420", position="acquisition"),
        )
    # add them to their table
    for member in people: db.insert(member)

    # make some books
    book = bizbook.schema.Book
    books = (
        book.pyre_immutable(
            id="BU1032",
            title="The busy executive's database guide",
            category="business", publisher="1389", date="1985/06/12",
            price=19.99, advance=5000,
            description="""
            An overview of available database systems with an emphasis on business
            applications. Illustrated."
            """),
        book.pyre_immutable(
            id="BU1111",
            title="Cooking with computers: surreptitious balance sheets",
            category="business", publisher="1389", date="1985/06/09",
            price=11.95, advance=5000,
            description="""
            Helpful hints on how to use your electronic resource to best advantage
            """),
        book.pyre_immutable(
            id="BU2075",
            title="You can combat computer stress",
            category="business", publisher="0736", date="1985/06/30",
            price=2.99, advance=10125,
            description="""
            The latest medical and psychological techniques for living with the electronic
            office. Easy to understand explanations.
            """),

        book.pyre_immutable(
            id="BU7832",
            title="Straight talk about computers",
            category="business", publisher="1389", date="1985/06/22",
            price=19.99, advance=5000,
            description="""
            Annotated analysis of what computers can do for you: a no-hype guide for the
            critical user.
            """),
        book.pyre_immutable(
            id="MC2222",
            title="Silicon Valley gastronomic treats",
            category="cookbook", publisher="0877", date="1985/06/09",
            price=19.99, advance=0,
            description="""
            Favorite recipes for quick, easy and elegant meals, tried and tested by people who
            never have time to eat, let alone cook.
            """),
        book.pyre_immutable(
            id="MC3021",
            title="The gourmet microwave",
            category="cookbook", publisher="0877", date="1985/06/18",
            price=2.99, advance=15000,
            description="""
            Traditional French gourmet recipes adapted for modern microwave cooking.
            """),
        book.pyre_immutable(
            id="MC3026",
            title="The psychology of computer cooking",
            publisher="0877",
            ),
        book.pyre_immutable(
            id="PC1035",
            title="But is it user-friendly?",
            category="computing", publisher="1389", date="1985/06/30",
            price=22.95, advance=7000,
            description="""
            A survey of software for the naïve user, focusing on the 'friendliness' of each.
            """),

        book.pyre_immutable(
            id="PC8888",
            title="Secrets of Silicon Valley",
            category="computing", publisher="1389", date="1985/06/12",
            price=20, advance=8000,
            description="""
            Muckraking reporting by two courageous women on the world's largest computer
            software and hardware manufacturers.
            """),
        book.pyre_immutable(
            id="PC9999",
            title="Net etiquette",
            category="computing", publisher="1389",
            description="""
            A must-read for computer conference debutantes
            """),
        book.pyre_immutable(
            id="PS1372",
            title="Computer-phobic and on-phobic individuals: behavior variations",
            category="psychology", publisher="0736", date="1985/10/21",
            price=21.59, advance=7000,
            description="""
            A must for the specialist, this book examines the difference between those who hate
            and fear computers and those who think they are swell.
            """),
        book.pyre_immutable(
            id="PS2091",
            title="Is anger the enemy?",
            category="psychology", publisher="0736", date="1985/06/15",
            price=10.95, advance=2275,
            description="""
            Carefully researched study of the effects of strong emotions on the body. Metabolic
            charts included.
            """),
        book.pyre_immutable(
            id="PS2106",
            title="Life without fear",
            category="psychology", publisher="0736", date="1985/10/05",
            price=7, advance=6000,
            description="""
            New exercise, meditation, and nutritional techniques that can reduce the shock of
            daily interactions. Popular audience. Sample menus included, exercise video
            available separately.
            """),

        book.pyre_immutable(
            id="PS3333",
            title="Prolonged data deprivation: four case studies",
            category="psychology", publisher="0736", date="1985/06/12",
            price=19.99, advance=2000,
            description="""
            What happens when the data runs dry? Searching evaluations of information-shortage
            effects on heavy users.
            """),
        book.pyre_immutable(
            id="PS7777",
            title="Emotional security: a new algorithm",
            category="psychology", publisher="0736", date="1985/06/12",
            price=7.99, advance=4000,
            description="""
            Protecting yourself and your loved ones from undue emotional stress in the modern
            world. Use of computer and nutritional aids emphasized.
            """),
        book.pyre_immutable(
            id="TC3218",
            title="Onions, leeks and garlic: cooking secrets of the Mediterranean",
            category="cookbook", publisher="0877", date="1985/10/21",
            price=20.95, advance=7000,
            description="""
            Profusely illustrated in color, this makes a wonderful gift book for a
            cuisine-oriented friend.
            """),

        book.pyre_immutable(
            id="TC7777",
            title="Sushi, Anyone?",
            category="cookbook", publisher="0877", date="1985/06/12",
            price=14.99, advance=8000,
            description="""
            Detailed instructions on improving your position in life by learning how to make
            authentic Japanese sushi in your spare time. 5-10% increase in number of friends
            per recipe reported from beta test.
            """),
        book.pyre_immutable(
            id="TC4203",
            title="Fifty years in Buckingham Palace kitchens",
            category="cookbook", publisher="0877", date="1985/06/12",
            price=11.95, advance=4000,
            description="""
            More anecdotes from the Queen's favorite cook describing life among English
            royalty. Recipes, techniques, tender vignettes.
            """),
        )
    # add them to their table
    for book in books: db.insert(book)

    # authors
    author = bizbook.schema.Author
    authors = (
        author.pyre_immutable(author="173-32-1176", book="PS3333", ordinal=1, share=1.0),
        author.pyre_immutable(author="213-46-8915", book="BU1032", ordinal=2, share=0.4),
        author.pyre_immutable(author="213-46-8915", book="BU2075", ordinal=1, share=1.0),
        author.pyre_immutable(author="238-95-7766", book="PC1035", ordinal=1, share=1.0),
        author.pyre_immutable(author="267-41-2394", book="BU1111", ordinal=2, share=0.4),
        author.pyre_immutable(author="267-41-2394", book="TC7777", ordinal=2, share=0.3),
        author.pyre_immutable(author="274-80-9391", book="BU7832", ordinal=1, share=1.0),
        author.pyre_immutable(author="409-56-7008", book="BU1032", ordinal=1, share=0.6),
        author.pyre_immutable(author="427-17-2319", book="PC8888", ordinal=1, share=0.5),
        author.pyre_immutable(author="472-27-2349", book="TC7777", ordinal=3, share=0.3),
        author.pyre_immutable(author="486-29-1786", book="PC9999", ordinal=1, share=1.0),
        author.pyre_immutable(author="486-29-1786", book="PS7777", ordinal=1, share=1.0),
        author.pyre_immutable(author="648-92-1872", book="TC4203", ordinal=1, share=1.0),
        author.pyre_immutable(author="672-71-3249", book="TC7777", ordinal=1, share=0.4),
        author.pyre_immutable(author="712-45-1867", book="MC2222", ordinal=1, share=1.0),
        author.pyre_immutable(author="722-51-5454", book="MC3021", ordinal=1, share=0.75),
        author.pyre_immutable(author="724-80-9391", book="BU1111", ordinal=1, share=0.6),
        author.pyre_immutable(author="724-80-9391", book="PS1372", ordinal=2, share=0.25),
        author.pyre_immutable(author="756-30-7391", book="PS1372", ordinal=1, share=0.75),
        author.pyre_immutable(author="807-91-6654", book="TC3218", ordinal=1, share=1.0),
        author.pyre_immutable(author="846-92-7186", book="PC8888", ordinal=2, share=0.5),
        author.pyre_immutable(author="899-46-2035", book="MC3021", ordinal=2, share=0.25),
        author.pyre_immutable(author="899-46-2035", book="PS2091", ordinal=2, share=0.5),
        author.pyre_immutable(author="998-72-3567", book="PS2091", ordinal=1, share=0.5),
        author.pyre_immutable(author="998-72-3567", book="PS2106", ordinal=1, share=1.0),
        )
    # add them to their table
    for author in authors: db.insert(author)

    # associate staff with books
    editor = bizbook.schema.Editor
    editors = (
        editor.pyre_immutable(editor="321-55-8906", book="BU1032", ordinal=2),
        editor.pyre_immutable(editor="321-55-8906", book="BU1111", ordinal=2),
        editor.pyre_immutable(editor="321-55-8906", book="BU2075", ordinal=3),
        editor.pyre_immutable(editor="321-55-8906", book="BU7832", ordinal=2),
        editor.pyre_immutable(editor="321-55-8906", book="PC1035", ordinal=2),
        editor.pyre_immutable(editor="321-55-8906", book="PC8888", ordinal=2),
        editor.pyre_immutable(editor="777-02-9831", book="PC1035", ordinal=3),
        editor.pyre_immutable(editor="777-02-9831", book="PC8888", ordinal=3),
        editor.pyre_immutable(editor="826-11-9034", book="BU2075", ordinal=2),
        editor.pyre_immutable(editor="826-11-9034", book="PS1372", ordinal=2),
        editor.pyre_immutable(editor="826-11-9034", book="PS2091", ordinal=2),
        editor.pyre_immutable(editor="826-11-9034", book="PS2106", ordinal=2),
        editor.pyre_immutable(editor="826-11-9034", book="PS3333", ordinal=2),
        editor.pyre_immutable(editor="826-11-9034", book="PS7777", ordinal=2),
        editor.pyre_immutable(editor="885-23-9140", book="MC2222", ordinal=2),
        editor.pyre_immutable(editor="885-23-9140", book="MC3021", ordinal=2),
        editor.pyre_immutable(editor="885-23-9140", book="TC3218", ordinal=2),
        editor.pyre_immutable(editor="885-23-9140", book="TC4203", ordinal=2),
        editor.pyre_immutable(editor="885-23-9140", book="TC7777", ordinal=2),
        editor.pyre_immutable(editor="943-88-7920", book="BU1032", ordinal=1),
        editor.pyre_immutable(editor="943-88-7920", book="BU1111", ordinal=1),
        editor.pyre_immutable(editor="943-88-7920", book="BU2075", ordinal=1),
        editor.pyre_immutable(editor="943-88-7920", book="BU7832", ordinal=1),
        editor.pyre_immutable(editor="943-88-7920", book="PC1035", ordinal=1),
        editor.pyre_immutable(editor="943-88-7920", book="PC8888", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="MC2222", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="MC3021", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="PS1372", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="PS2091", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="PS2106", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="PS3333", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="PS7777", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="TC3218", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="TC4203", ordinal=1),
        editor.pyre_immutable(editor="993-86-0420", book="TC7777", ordinal=1),
        )
    # add them to their table
    for editor in editors: db.insert(editor)

    # make some invoices
    invoice = bizbook.schema.Invoice
    invoices = (
        invoice.pyre_immutable(id="I01", client="7066", po="QA7442.3", date="1985/09/13"),
        invoice.pyre_immutable(id="I02", client="7067", po="D4482", date="1985/09/14"),
        invoice.pyre_immutable(id="I03", client="7131", po="N914008", date="1985/09/14"),
        invoice.pyre_immutable(id="I04", client="7131", po="N914014", date="1985/09/14"),
        invoice.pyre_immutable(id="I05", client="8042", po="423LL922", date="1985/09/14"),
        invoice.pyre_immutable(id="I06", client="8042", po="423LL930", date="1985/09/14"),
        invoice.pyre_immutable(id="I07", client="6380", po="722a", date="1985/09/13"),
        invoice.pyre_immutable(id="I08", client="6380", po="6871", date="1985/09/14"),
        invoice.pyre_immutable(id="I09", client="8042", po="P723", date="1988/03/11"),
        invoice.pyre_immutable(id="I19", client="7896", po="X999", date="1988/02/21"),
        invoice.pyre_immutable(id="I10", client="7896", po="QQ2299", date="1987/10/28"),
        invoice.pyre_immutable(id="I11", client="7896", po="TQ456", date="1987/12/12"),
        invoice.pyre_immutable(id="I12", client="8042", po="QA879.1", date="1987/05/22"),
        invoice.pyre_immutable(id="I13", client="7066", po="A2976", date="1987/05/24"),
        invoice.pyre_immutable(id="I14", client="7131", po="P3087a", date="1987/05/29"),
        invoice.pyre_immutable(id="I15", client="7067", po="P2121", date="1987/06/15"),
        )
    # add them to their table
    for invoice in invoices: db.insert(invoice)

    # make some line items
    item = bizbook.schema.InvoiceItem
    items = (
        item.pyre_immutable(invoice="I01", book="PS2091", ordered=75,shipped=75, date="1985/09/15"),
        item.pyre_immutable(invoice="I02", book="PS2091", ordered=10,shipped=10, date="1985/09/15"),
        item.pyre_immutable(invoice="I03", book="PS2091", ordered=20,shipped=20, date="1985/09/18"),
        item.pyre_immutable(invoice="I04", book="MC3021", ordered=25,shipped=20, date="1985/09/18"),
        item.pyre_immutable(invoice="I05", book="MC3021", ordered=15,shipped=15, date="1985/09/14"),
        item.pyre_immutable(invoice="I06", book="BU1032", ordered=10,shipped=3, date="1985/09/22"),
        item.pyre_immutable(invoice="I07", book="PS2091", ordered=3,shipped=3, date="1985/09/20"),
        item.pyre_immutable(invoice="I08", book="BU1032", ordered=5,shipped=5, date="1985/09/14"),
        item.pyre_immutable(invoice="I09", book="BU1111", ordered=25,shipped=5, date="1988/03/28"),
        item.pyre_immutable(invoice="I19", book="BU2075", ordered=35,shipped=35, date="1988/03/15"),
        item.pyre_immutable(invoice="I10", book="BU7832", ordered=15,shipped=15, date="1987/10/29"),
        item.pyre_immutable(invoice="I11", book="MC2222", ordered=10,shipped=10, date="1988/01/12"),
        item.pyre_immutable(invoice="I12", book="PC1035", ordered=30,shipped=30, date="1987/03/24"),
        item.pyre_immutable(invoice="I13", book="PC8888", ordered=50,shipped=50, date="1987/03/24"),
        item.pyre_immutable(invoice="I14", book="PS1372", ordered=20,shipped=20, date="1987/03/29"),
        item.pyre_immutable(invoice="I14", book="PS2106", ordered=25,shipped=25, date="1987/04/29"),
        item.pyre_immutable(invoice="I14", book="PS3333", ordered=15,shipped=10, date="1987/05/29"),
        item.pyre_immutable(invoice="I14", book="PS7777", ordered=25,shipped=25, date="1987/06/13"),
        item.pyre_immutable(invoice="I15", book="TC3218", ordered=40,shipped=40, date="1987/06/15"),
        item.pyre_immutable(invoice="I15", book="TC4203", ordered=20,shipped=20, date="1987/05/30"),
        item.pyre_immutable(invoice="I15", book="TC7777", ordered=20,shipped=10, date="1987/06/17"),
        )
    # add them to their table
    for item in items: db.insert(item)

    db.connection.commit()

    # and return the component
    return db


# main
if __name__ == "__main__":
    test()


# end of file
