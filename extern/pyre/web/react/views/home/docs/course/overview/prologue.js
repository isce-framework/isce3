// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'

// support
import { Pyre, Paragraph } from 'widgets/doc'

// locals
import styles from './styles'

// declaration
const Prologue = () => (
    <div>
        <Paragraph>
            Research codes are peculiar beasts. They are typically born to parents that are far
            too young and inexperienced to take proper care of them. Their early days are
            sickly and even their authors do not expect them to live any longer than the next
            paper, definitely not past the completion of their thesis. They grow up in
            haphazard ways, reflecting the evolution of their care provider's understanding of
            some research problem, end up having far too many appendages sticking out of all
            the wrong places, and display very few signs of any organizing principle, let alone
            design.  Yet, many of them outlive their parent's wildest expectations (or fears),
            have long, productive lives, and become focal points of entire research
            communities. Hated by all, but also used by all.
        </Paragraph>

        <Paragraph>
            The good ones have buried in them precious intellectual capital; that's the secret
            of their longevity. The reason they are constantly on the
            some-day-I-will-rewrite-this list is almost always the lack of enough structure so
            that successive generations of foster parents can maintain and evolve the code. The
            missing structure goes by the name <em>modern software engineering practices</em>,
            and it's not on the list of skills that graduate students of respectable
            institutions are supposed to have.
        </Paragraph>

        <Paragraph>
            Pyre, the software architecture described in this paper, is an attempt to bring
            state of the art software design practices to scientific computing. The goal is to
            provide a strong skeleton on which to build scientific codes by steering the
            implementation towards usability and maintainability. It's not a substitute for the
            intelligence, experience and effort necessary to write robust, hardened
            software. But by encouraging you to ask the right design questions and make the
            right practices part of your software cycle, you should experience a dramatic
            improvement in the quality of code you write.
        </Paragraph>

        <Paragraph>
            You will still need to shop around for a source control system and find a scalable
            way to build your software on multiple platforms. You will need to find a good
            solution for writing documentation, maintain and run test suites, and track bugs
            and feature requests. If you are really ambitious, you need a release management
            solution. Most people hope they can get away without all this overhead, but that's
            just the mild form of delusion that comes from not knowing what you are in for.
            Writing software can easily degenerate into a chaotic practice, with small changes
            having potentially unbounded effects, even when it's only you that's doing the
            coding. The passage of time has a way of introducing interesting complexity in
            software systems.
        </Paragraph>

        <Paragraph>
            In the next few sections, I will show you how to turn a throw-away script into a
            usable application. After a brief introduction to the method, we will start by
            writing a naïve implementation of Monte Carlo integration in both python and
            C++. Then, we will evolve the code by introducing object oriented concepts that
            will help us improve and document the design of the code. The last evolutionary
            step will be to cast the design as a collection of reusable components. Once we
            have that, we will explore user interfaces, parallelism and more.
        </Paragraph>

    </div>
)

//   publish
export default Prologue

// end of file
