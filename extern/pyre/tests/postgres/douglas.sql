-- -*- coding: utf-8 -*-
--
-- michael a.g. aïvázis
-- orthologue
-- (c) 1998-2019 all rights reserved
--

-- sample database from "Postgresql" by Douglas & Douglas

-- table declarations

CREATE TABLE "customers" (
  "customer_id" integer unique not null,
  "customer_name" character varying(50) not null,
  "phone" character(8) null,
  "birth_date" date null,
  "balance" decimal(7,2)
);


CREATE TABLE "tapes" (
  "tape_id" character(8) not null,
  "title" character varying(80) not null,
  "duration" interval
);


CREATE TABLE "rentals" (
  "tape_id" character(8) not null,
  "rental_date" date not null,
  "customer_id" integer not null
);


-- data

INSERT INTO customers
  VALUES
    (3, 'Panky, Henry', '555-1221', '1968-01-21', 0.00),
    (1, 'Jones, Henry', '555-1212', '1970-10-10', 0.00),
    (4, 'Wonderland, Alice N.', '555-1122', '1969-03-05', 3.00),
    (2, 'Rubin, William', '555-2211', '1972-07-10', 15.00);

INSERT INTO tapes
  VALUES
    ('AB-12345', 'The Godfather'),
    ('AB-67472', 'The Godfather'),
    ('MC-68873', 'Casablanca'),
    ('OW-41221', 'Citizen Kane'),
    ('AH-54706', 'Rear Window');

INSERT INTO rentals
  VALUES
    ('AB-12345', '2001-11-25', 1),
    ('AB-67472', '2001-11-25', 3),
    ('OW-41221', '2001-11-25', 1),
    ('MC-68873', '2001-11-20', 3);

-- end of file
