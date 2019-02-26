#include <cstring> // std::memcpy
#include <random> // std::mt19937, std::uniform_real_distribution
#include <utility> // std::swap

#include "ICU.h" // ICU, idx2_t, offset2_t
#include "SearchTable.h" // SearchTable

namespace isce { namespace unwrap { namespace icu {

template<typename T, class URNG>
void permute(
    T * shuffled,
    const T * input, 
    const size_t size, 
    URNG & generator)
{
    // Handle zero residues case.
    if (size == 0) { return; }

    // Copy to output buffer.
    std::memcpy(shuffled, input, size * sizeof(T));

    // Permute in-place using Fisher-Yates algorithm.
    auto dist = std::uniform_real_distribution<double>(0.0, 1.0);
    for (size_t n = size-1; n > 0; --n)
    {
        auto i = size_t(double(n) * dist(generator));
        std::swap(shuffled[i], shuffled[n]);
    }
}

class Twig
{
public:
    // Constructor
    Twig(const size_t length, const size_t width);
    // Destructor
    ~Twig();
    // Add node to end of twig.
    void push(const idx2_t node);
    // Access node at specified position.
    const idx2_t & operator[](const size_t pos) const;
    // Check if twig contains the specified node.
    bool contains(const idx2_t & node) const;
    // Clear twig contents.
    void clear();
    // Get number of nodes on twig.
    size_t size() const;
private:
    // Twig nodes (residues or neutrons)
    std::vector<idx2_t> _nodes;
    // Mask of nodes on twig
    bool * _contains;
    // Tile width
    size_t _width;
};

Twig::Twig(const size_t length, const size_t width)
:
    _width(width)
{
    // Pre-allocate storage for a large number of nodes.
    const size_t tilesize = length * width;
    _nodes.reserve(tilesize/20);

    // Init mask of nodes on twig.
    _contains = new bool[tilesize];
    for (size_t i = 0; i < tilesize; ++i) { _contains[i] = false; }
}

Twig::~Twig()
{
    delete[] _contains;
}

void Twig::push(const idx2_t node)
{
    _nodes.push_back(node);

    size_t inode = node[1] * _width + node[0];
    _contains[inode] = true;
}

const idx2_t & Twig::operator[](const size_t pos) const
{
    return _nodes[pos];
}

bool Twig::contains(const idx2_t & node) const
{
    size_t inode = node[1] * _width + node[0];
    return _contains[inode];
}

void Twig::clear()
{
    for (size_t t = 0; t < _nodes.size(); ++t)
    {
        idx2_t node = _nodes[t];
        size_t inode = node[1] * _width + node[0];
        _contains[inode] = false;
    }
    _nodes.clear();
}

inline size_t Twig::size() const { return _nodes.size(); }

void branchcutHoriz(
    bool * tree,
    size_t i1,
    size_t i2, 
    size_t j,
    const size_t width)
{
    if (i1 > i2) { std::swap(i1, i2); }
    for (size_t i = i1; i <= i2; ++i) { tree[j * width + i] = true; }
}

void branchcutVert(
    bool * tree,
    size_t i, 
    size_t j1,
    size_t j2,
    const size_t width)
{
    if (j1 > j2) { std::swap(j1, j2); }
    for (size_t j = j1; j <= j2; ++j) { tree[j * width + i] = true; }
}

void branchcutShallow(
    bool * tree,
    size_t i1, size_t j1,
    size_t i2, size_t j2,
    const size_t width)
{
    if (i1 > i2)
    {
        std::swap(i1, i2);
        std::swap(j1, j2);
    }

    // Draw a line using Bresenham's algorithm.
    int di = i2 - i1;
    int dj = j2 - j1;
    int inc = 1;
    if (j1 > j2)
    {
        dj = j1 - j2;
        inc = -1;
    }

    size_t i = i1;
    size_t j = j1;
    tree[j * width + i] = true;

    int d = 2*dj - di;
    for (i = i1+1; i <= i2; ++i)
    {
        if (d > 0)
        {
            j += inc;
            d -= 2*di;
        }
        d += 2*dj;
        tree[j * width + i] = true;
    }
}

void branchcutSteep(
    bool * tree,
    size_t i1, size_t j1,
    size_t i2, size_t j2,
    const size_t width)
{
    if (j1 > j2)
    {
        std::swap(i1, i2);
        std::swap(j1, j2);
    }

    // Draw a line using Bresenham's algorithm.
    int di = i2 - i1;
    int dj = j2 - j1;
    int inc = 1;
    if (i1 > i2)
    {
        di = i1 - i2;
        inc = -1;
    }

    size_t i = i1;
    size_t j = j1;
    tree[j * width + i] = true;

    int d = 2*di - dj;
    for (j = j1+1; j <= j2; ++j)
    {
        if (d > 0)
        {
            i += inc;
            d -= 2*dj;
        }
        d += 2*di;
        tree[j * width + i] = true;
    }
}

void branchcut(
    bool * tree,
    size_t i1, size_t j1,
    size_t i2, size_t j2,
    const size_t width)
{
    const int di = (i2 > i1) ? (i2 - i1) : (i1 - i2);
    const int dj = (j2 > j1) ? (j2 - j1) : (j1 - j2);

    if      (dj == 0)  { branchcutHoriz(tree, i1, i2, j1, width); }
    else if (di == 0)  { branchcutVert(tree, i1, j1, j2, width); }
    else if (di >= dj) { branchcutShallow(tree, i1, j1, i2, j2, width); }
    else               { branchcutSteep(tree, i1, j1, i2, j2, width); }
}

void growTwig(
    bool * tree,
    Twig & twig,
    bool * visited,
    const SearchTable & searchtable,
    const signed char * charge, 
    const bool * neut, 
    const idx2_t & root,
    const size_t length, 
    const size_t width,
    const int maxbranchlen)
{
    // Reset twig.
    twig.clear();

    // Init total charge on twig.
    int twigcharge = 0;

    // Add root residue to twig.
    size_t iroot = root[1] * width + root[0];
    visited[iroot] = true;
    twigcharge += charge[iroot];
    twig.push(root);

    // Iteratively increase search radius up to max branch length.
    for (int l = 1; l <= maxbranchlen; ++l)
    {
        size_t nsearchpts = searchtable.numPtsInEllipse(l);

        // Loop over twig.
        for (size_t t = 0; t < twig.size(); ++t)
        {
            idx2_t node = twig[t];

            // Search for unvisited residues or neutrons within the current 
            // search radius of the current node.
            for (size_t s = 0; s < nsearchpts; ++s)
            {
                offset2_t off = searchtable[s];

                // Check for out-of-bounds (avoiding underflow for 
                // unsigned + signed). If out-of-bounds, make horizontal or 
                // vertical branch cut to nearest edge and discharge twig.
                if (off[0] < 0 && node[0] < -off[0])
                { 
                    branchcutHoriz(tree, 0, node[0], node[1], width); 
                    return; 
                }
                else if (off[0] > 0 && node[0] + off[0] > width-1)
                { 
                    branchcutHoriz(tree, node[0], width-1, node[1], width); 
                    return; 
                }
                else if (off[1] < 0 && node[1] < -off[1])
                {
                    branchcutVert(tree, node[0], 0, node[1], width);
                    return;
                }
                else if (off[1] > 0 && node[1] + off[1] > length-1)
                {
                    branchcutVert(tree, node[0], node[1], length-1, width);
                    return;
                }

                // Check if search point is residue or neutron not already on twig.
                idx2_t newnode = {node[0] + off[0], node[1] + off[1]};
                size_t inewnode = newnode[1] * width + newnode[0];
                if ((charge[inewnode] != 0 || neut[inewnode]) && 
                    !twig.contains(newnode))
                {
                    // Make branch cut to new node.
                    branchcut(tree, node[0], node[1], newnode[0], newnode[1], width);

                    // Check if the new node discharges the twig (if the node 
                    // was previously visited, it is considered neutralized).
                    if (!visited[inewnode])
                    {
                        visited[inewnode] = true;

                        twigcharge += charge[inewnode];
                        if (twigcharge == 0) { return; }
                    }

                    // Add new node to twig.
                    twig.push(newnode);
                }
            }
        }
    }
}

void ICU::growTrees(
    bool * tree, 
    const signed char * charge,
    const bool * neut,
    const size_t length, 
    const size_t width, 
    const unsigned int seed)
{
    // Init branch cuts.
    const size_t tilesize = length * width;
    for (size_t i = 0; i < tilesize; ++i) { tree[i] = false; }

    // Get list of residue indices and residue count.
    auto resid = new idx2_t[tilesize];
    size_t nresid = 0;
    for (size_t j = 0; j < length; ++j)
    {
        for (size_t i = 0; i < width; ++i)
        {
            if (charge[j * width + i] != 0)
            {
                resid[nresid] = {i, j};
                ++nresid;
            }
        }
    }
    
    // Construct lookup table of search points in order of increasing distance.
    auto searchtable = SearchTable(_MaxBranchLen, _RatioDxDy);

    // Loop over tree realizations (final tree is the union of all realizations).
    #pragma omp parallel for shared(searchtable)
    for (int t = 0; t < _NumTrees; ++t)
    {
        // Twig keeps track of residues/nodes on the un-discharged 
        // (currently growing) part of the tree
        auto twig = Twig(length, width);

        // Mask of nodes visited by any twig in current tree
        auto visited = new bool[tilesize];
        for (size_t i = 0; i < tilesize; ++i) { visited[i] = false; }

        // Loop over a unique random permutation of the residue list.
        auto residshfl = new idx2_t[nresid];
        auto generator = std::mt19937(seed + t);
        permute(residshfl, resid, nresid, generator);
        for (int r = 0; r < nresid; ++r)
        {
            // Skip residue if already visited.
            idx2_t root = residshfl[r];
            size_t iroot = root[1] * width + root[0];
            if (visited[iroot]) { continue; }

            // Grow twig from residue.
            growTwig(
                tree, twig, visited, searchtable, charge, neut, root, length, 
                width, _MaxBranchLen);
        }

        delete[] visited;
        delete[] residshfl;
    }

    delete[] resid;
}

} } }

