// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import { Link } from 'react-router-dom'

// support
import { Narrative, Page, Pyre, Section, Subsection, Paragraph } from 'widgets/doc'
// locals
import styles from './styles'
import PyreRepo from './pyre-repo'
import PyreBuild from './pyre-build'
import PyreInstall from './pyre-install'
import ConfigRepo from './config-repo'
import ConfigInstall from './config-install'
import ConfigDeveloper from './config-developer'

// declaration
const Install = () => (
    <Page>
        <Narrative>
            <Section id="install" style={styles} title="installation instructions">

                <Subsection title="For the impatient --- and those with commitment issues">
                    <Paragraph>
                        If you just want to explore <Pyre/> and don't need the C++ bindings to
                        the supported infrastructure, you can <a
                        href="http://pyre.orthologue.com/pyre-1.0-boot.zip">download</a> a ZIP
                        file that contains a minimal but functional <Pyre/> installation.
                    </Paragraph>

                    <Paragraph>
                        Some browsers unzip downloaded files for you. This feature may be very
                        useful in general, but it doesn't work for our purposes. You may have
                        to <em>right-click</em> on the link and pick an appropriate option from
                        the context menu.
                    </Paragraph>

                    <Paragraph>
                        Save this file somewhere on your filesystem and add the full path to
                        your <code>PYTHONPATH</code> environment variable.  Don't forget to
                        include the actual name of the file, including
                        the <code>.zip</code> extension, since the interpreter treats ZIP
                        files as if they were
                        actual folders with python code.
                    </Paragraph>

                </Subsection>

                <Subsection title="binaries">
                    <Paragraph>
                        You can download pre-built binaries of the latest stable release
                        of <Pyre/>.  The binaries assume that you have installed compatible
                        versions of all the packages on which <Pyre/> depends, a requirement
                        that is often a lot trickier than it sounds. You may want to try these
                        first, before resorting to building <Pyre/> from source.
                    </Paragraph>

                    <Paragraph>
                        Currently, we build binaries for macOS high sierra for both
                        Python <a href="/pyre-1.0.cpython-36m-darwin.zip">3.6</a> and <a
                        href="/pyre-1.0.cpython-37m-darwin.zip">3.7</a>.
                        For linux, we build binaries for ubuntu 18.04 for Python <a
                        href="/pyre-1.0.cpython-36m-x86_64-linux-gnu.zip">3.6</a> and <a
                        href="/pyre-1.0.cpython-37m-x86_64-linux-gnu.zip">3.7</a>.
                        If you would like to see another platform added to this list, please <Link
                        to="/contact">let us know</Link>.
                    </Paragraph>

                </Subsection>

                <Subsection title="Getting the source code">
                    <Paragraph>
                        When you are ready to dig a little deeper, take a look at
                        the <Pyre/> source code.  There are a couple of options.  You can <Link
                        to="/pyre-1.0-source.tar.bz2">get</Link> a TAR file with the source
                        code of the last stable release.  Alternatively, you can prepare an
                        area on your filesystem and pull from the BZR repository:
                    </Paragraph>

                    <PyreRepo/>

                </Subsection>

                <Subsection title="Building the source code">

                    <Paragraph>
                        Building a <Pyre/> installation from its source requires access
                        to <code>config</code>, its build system. You can
                        pull <code>config</code> from its BZR repository:
                    </Paragraph>

                    <ConfigRepo/>

                    <Paragraph>
                        <code>config</code> is a rather powerful configuration management tool
                        built on top of GNU <code>make</code>. In order to simplify access to
                        the tool, you should add the following aliases to the startup files for
                        your shell:
                    </Paragraph>

                    <ConfigInstall/>

                    <Paragraph>
                        The aliases above are for a BASH compatible shell; please adjust to
                        your environment. Note that we have assumed you have
                        pulled <code>config</code> into <code>~/tmp</code>; please make the
                        necessary adjustments if you chose a different location for the
                        <code>config</code> repository.
                    </Paragraph>

                    <Paragraph>
                        You can now switch to the <Pyre/> source directory and build. If all
                        goes well, you will see a few screenfuls go by as <Pyre/> gets built
                        and tested. If you see a failure, <Link to="/contact">let us
                        know</Link>.
                    </Paragraph>

                    <PyreBuild/>

                    <Paragraph>
                        If you would like to control where <Pyre/> gets installed, there are a
                        couple of extra steps. Create the directory <code>~/.mm</code>, go
                        there, and create a file with the same name as your user name and with
                        the <code>.py</code> extension. For example, if your username
                        is <code>mga</code>
                    </Paragraph>

                    <PyreInstall/>

                    <Paragraph>
                        Here are the contents of the developer configuration file
                    </Paragraph>

                    <ConfigDeveloper/>

                    <Paragraph>
                        Please remember to use your own username as the name of the file. There
                        is nothing magical about the directory choices in this file. You can
                        put the temporary build files and the <Pyre/> installation anywhere you
                        have permissions to write.
                    </Paragraph>

                    <Paragraph>
                        We are always happy to listen to your feedback and help. Do not
                        hesitate to <Link to="/contact">get in touch</Link> if you have
                        something to share.
                    </Paragraph>

                </Subsection>

            </Section>
        </Narrative>
    </Page>
)

//   publish
export default Install

// end of file
