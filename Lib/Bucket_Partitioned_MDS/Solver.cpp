#include "Utils.h"
#include "Bucket_Partitioned_MDS.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <omp.h>

#ifdef BPMDS_USE_CUDA
#include "Cuda_Hotpath.h"
#endif

namespace Bucket_Partitioned_MDS
{
    void Solver::create_buckets(
        const CVRP& cvrp, 
        std::vector <std::vector <node_t>>& buckets) const 
    {
        const int num_buckets = static_cast<int>(buckets.size());
        const int N = cvrp.size();
        if(num_buckets == 1)
        {
            buckets[0].reserve(N);
            for(node_t u = 0; u < N; u++)
            {
                buckets[0].push_back(u);
            }
            return;
        }

        const node_t depot = cvrp.depot();
        for(int i = 0; i < num_buckets; i++)
        {
            buckets[i].push_back(depot);
            buckets[i].reserve(static_cast<size_t>(N) / static_cast<size_t>(num_buckets) + 8);
        }

        const Point& depot_pt = cvrp[depot];
        for(node_t u = 1; u < N; u++)
        {
            const double dx = cvrp[u].x - depot_pt.x;
            const double dy = cvrp[u].y - depot_pt.y;
            if (dx == 0.0 && dy == 0.0)
            {
                buckets[0].push_back(u);
                continue;
            }
            double deg = std::atan2(dy, dx) * 180.0 / PI;
            if (deg < 0.0)
            {
                deg += 360.0;
            }
            int idx = static_cast<int>(deg / alpha);
            if (idx >= num_buckets)
            {
                idx = num_buckets - 1;
            }
            if (idx < 0)
            {
                idx = 0;
            }
            buckets[idx].push_back(u);
        }
    }

    void Solver::get_bucket(
        const CVRP&             cvrp, 
        const double            start_angle, 
        const double            end_angle, 
        std::vector <node_t>&   bucket) const
    {
        const int N = cvrp.size();
        const node_t depot = cvrp.depot();
        bucket.push_back(depot); 

        Unit_Vector_2D xaxis(1, 0);        
        Unit_Vector_2D start_vector(xaxis, start_angle);
        Unit_Vector_2D end_vector(xaxis, end_angle);

        for(node_t u = 1; u < N; u++)  
        {
            Unit_Vector_2D vec(cvrp[depot], cvrp[u]);
            if(vec.is_in_between(start_vector, end_vector))
            {
                bucket.push_back(u);
            }
        }
    }

    namespace {
        int env_int(const char* key, int fallback)
        {
            const char* v = std::getenv(key);
            if (!v || !*v) {
                return fallback;
            }
            return std::atoi(v);
        }

        bool cuda_enabled()
        {
#ifdef BPMDS_USE_CUDA
            const char* v = std::getenv("BPMDS_CUDA");
            if (v && (v[0] == '0') && v[1] == '\0') {
                return false;
            }
            return bpmds_cuda_available() != 0;
#else
            return false;
#endif
        }
    } // namespace

    void Solver::construct_mst(
        const CVRP&                         cvrp,
        const std::vector <node_t>&         bucket,
        std::vector <std::vector <node_t>>& adj) const
    {
        // Dense Prim on squared lengths (sqrt is monotone; ties use smallest index).
        // Memory is O(bucket) — never a full-instance N×N matrix.
        const int n = static_cast<int>(bucket.size());
        adj.assign(n, {});
        if (n <= 1)
        {
            return;
        }

        std::vector<double> xs(n), ys(n);
        for (int i = 0; i < n; ++i)
        {
            xs[i] = cvrp[bucket[i]].x;
            ys[i] = cvrp[bucket[i]].y;
        }

#ifdef BPMDS_USE_CUDA
        const int prim_min = env_int("BPMDS_CUDA_PRIM_MIN", 512);
        if (cuda_enabled() && n >= prim_min)
        {
            std::vector<int> parent(n, -1);
            if (bpmds_cuda_prim_mst(xs.data(), ys.data(), n, parent.data()) == 0)
            {
                bool ok = true;
                for (int i = 1; i < n; ++i)
                {
                    const int p = parent[i];
                    if (p < 0 || p >= n)
                    {
                        ok = false;
                        break;
                    }
                    adj[p].push_back(i);
                    adj[i].push_back(p);
                }
                int edges = 0;
                for (int i = 0; i < n; ++i)
                {
                    edges += static_cast<int>(adj[i].size());
                }
                if (ok && edges == 2 * (n - 1))
                {
                    return;
                }
                adj.assign(n, {});
            }
        }
#endif

        constexpr double kInf = std::numeric_limits<double>::infinity();
        std::vector<double> best(n, kInf);
        std::vector<int> parent(n, -1);
        std::vector<char> in_tree(n, 0);

        in_tree[0] = 1;
        for (int j = 1; j < n; ++j)
        {
            const double dx = xs[0] - xs[j];
            const double dy = ys[0] - ys[j];
            best[j] = dx * dx + dy * dy;
            parent[j] = 0;
        }

        for (int added = 1; added < n; ++added)
        {
            int v = -1;
            double best_w = kInf;
            for (int j = 1; j < n; ++j)
            {
                if (!in_tree[j] && best[j] < best_w)
                {
                    best_w = best[j];
                    v = j;
                }
            }
            if (v < 0)
            {
                break;
            }

            in_tree[v] = 1;
            adj[parent[v]].push_back(v);
            adj[v].push_back(parent[v]);

            const double xv = xs[v];
            const double yv = ys[v];
            for (int w = 0; w < n; ++w)
            {
                if (in_tree[w])
                {
                    continue;
                }
                const double dx = xv - xs[w];
                const double dy = yv - ys[w];
                const double dw = dx * dx + dy * dy;
                if (dw < best[w])
                {
                    best[w] = dw;
                    parent[w] = v;
                }
            }
        }
    }

    distance_t Solver::get_route_distance(
        const CVRP&                 cvrp,
        const std::vector <node_t>& route) const
    {
        node_t prev_node = cvrp.depot();
        distance_t cost = 0;

        for(auto& node: route)
        {
            cost += cvrp.distance_fast(prev_node, node);
            prev_node = node;
        }
        cost += cvrp.distance_fast(prev_node, cvrp.depot());

        return cost;
    }

    void Solver::tsp_approx(
        const CVRP&             cvrp, 
        std::vector<node_t>&    cities, 
        std::vector<node_t>&    tour, 
        const int               ncities) const 
    {
        node_t i, j;
        node_t ClosePt = 0;
        distance_t CloseDist;

        std::copy(cities.begin(), cities.end() - 1, tour.begin() + 1);
        tour[0] = cities.back();

        for (i = 1; i < ncities; i++) 
        {
            distance_t ThisX = cvrp[tour[i - 1]].x;
            distance_t ThisY = cvrp[tour[i - 1]].y;
            CloseDist = DBL_MAX;

            for (j = ncities - 1;; j--) 
            {
                distance_t ThisDist = (cvrp[tour[j]].x - ThisX) * (cvrp[tour[j]].x - ThisX);
                if (ThisDist <= CloseDist) 
                {
                    ThisDist += (cvrp[tour[j]].y - ThisY) * (cvrp[tour[j]].y - ThisY);
                    if (ThisDist <= CloseDist) 
                    {
                        if (j < i)
                        {
                            break;
                        }
                        CloseDist = ThisDist;
                        ClosePt = j;
                    }
                }
            }

            auto temp = tour[i];
            tour[i] = tour[ClosePt];
            tour[ClosePt] = temp;
        }
    }

    std::vector<std::vector<node_t>> Solver::process_tsp_approx(
        const CVRP&                             cvrp, 
        const std::vector<std::vector<node_t>>& sol_routes) const 
    {
        std::vector<std::vector<node_t>> modifiedRoutes;
        const int num_routes = sol_routes.size();
        modifiedRoutes.reserve(sol_routes.size());

        for(int i = 0; i < num_routes; i++)
        {
            int sz = sol_routes[i].size();
            std::vector<node_t> cities = sol_routes[i];
            cities.push_back(cvrp.depot()); 
            std::vector<node_t> tour(sz + 1);
            tsp_approx(cvrp, cities, tour, sz + 1);

            std::vector<node_t> curr_route; 
            curr_route.reserve(sz);
            for (int kk = 1; kk < sz + 1; ++kk) 
            { 
                curr_route.push_back(tour[kk]); 
            } 
            modifiedRoutes.push_back(std::move(curr_route));
        }
        return modifiedRoutes;
    }

    void Solver::tsp_2OPT(
        const CVRP&             cvrp, 
        std::vector <node_t>&   cities, 
        std::vector <node_t>&   /*tour*/, 
        int                     ncities) const
    {
        // O(1) edge-length delta; kMaxPasses caps rare FP oscillation.
        const node_t depot = cvrp.depot();

#ifdef BPMDS_USE_CUDA
        const int opt_min = env_int("BPMDS_CUDA_2OPT_MIN", 128);
        if (cuda_enabled() && ncities >= opt_min)
        {
            const int N = cvrp.size();
            std::vector<double> cx(N), cy(N);
            for (int i = 0; i < N; ++i)
            {
                cx[i] = cvrp[i].x;
                cy[i] = cvrp[i].y;
            }
            if (bpmds_cuda_tsp_2opt(
                    cities.data(), ncities, cx.data(), cy.data(), N, depot)
                == 0)
            {
                return;
            }
        }
#endif

        int improve = 0;
        int passes = 0;
        constexpr int kMaxPasses = 32;

        while (improve < 2 && passes < kMaxPasses) 
        {
            ++passes;
            for (int i = 0; i < ncities - 1; i++) 
            {
                for (int k = i + 1; k < ncities; k++) 
                {
                    const node_t left  = (i == 0) ? depot : cities[i - 1];
                    const node_t right = (k + 1 == ncities) ? depot : cities[k + 1];
                    const node_t a = cities[i];
                    const node_t b = cities[k];

                    const double delta =
                        cvrp.distance_fast(left, b) + cvrp.distance_fast(a, right)
                        - cvrp.distance_fast(left, a) - cvrp.distance_fast(b, right);

                    if (delta < -1e-12) 
                    {
                        std::reverse(cities.begin() + i, cities.begin() + k + 1);
                        improve = 0;
                    }
                }
            }
            improve++;
        }
    }

    void Solver::tsp_or_opt(
        const CVRP& cvrp,
        std::vector<node_t>& cities) const
    {
        // Or-opt: move a consecutive segment of length L∈{1,2,3} (L=1 ≡ relocate).
        const int n = static_cast<int>(cities.size());
        if (n < 2)
        {
            return;
        }
        const node_t depot = cvrp.depot();
        auto pred = [&](int idx) -> node_t {
            return idx <= 0 ? depot : cities[idx - 1];
        };

        int passes = 0;
        constexpr int kMaxPasses = 16;
        bool improved = true;
        while (improved && passes < kMaxPasses)
        {
            improved = false;
            ++passes;
            const int nn = static_cast<int>(cities.size());
            for (int L = 1; L <= 3 && L < nn; ++L)
            {
                for (int i = 0; i + L <= nn; ++i)
                {
                    const node_t left = pred(i);
                    const node_t right = (i + L < nn) ? cities[i + L] : depot;
                    const node_t seg_first = cities[i];
                    const node_t seg_last = cities[i + L - 1];
                    const double remove_delta =
                        cvrp.distance_fast(left, right)
                        - cvrp.distance_fast(left, seg_first)
                        - cvrp.distance_fast(seg_last, right);

                    for (int p = 0; p <= nn - L; ++p)
                    {
                        // p = insert index in the route after removal.
                        if (p == i || p == i + 1)
                        {
                            continue;
                        }
                        int orig_before;
                        int orig_after;
                        if (p <= i)
                        {
                            orig_before = p - 1;
                            orig_after = p;
                        }
                        else
                        {
                            orig_before = p + L - 1;
                            orig_after = p + L;
                        }
                        const node_t ins_left =
                            (orig_before < 0) ? depot : cities[orig_before];
                        const node_t ins_right =
                            (orig_after >= nn) ? depot : cities[orig_after];

                        const double insert_delta =
                            cvrp.distance_fast(ins_left, seg_first)
                            + cvrp.distance_fast(seg_last, ins_right)
                            - cvrp.distance_fast(ins_left, ins_right);
                        if (remove_delta + insert_delta < -1e-12)
                        {
                            std::vector<node_t> seg(
                                cities.begin() + i, cities.begin() + i + L);
                            cities.erase(
                                cities.begin() + i, cities.begin() + i + L);
                            cities.insert(
                                cities.begin() + p, seg.begin(), seg.end());
                            improved = true;
                            break;
                        }
                    }
                    if (improved)
                    {
                        break;
                    }
                }
                if (improved)
                {
                    break;
                }
            }
        }
    }

    void Solver::tsp_3opt(
        const CVRP& cvrp,
        std::vector<node_t>& cities) const
    {
        // Limited 3-opt: reorder two interior segments (with optional reverses).
        // Caps route size so O(n^3)-ish search stays practical.
        const int n = static_cast<int>(cities.size());
        if (n < 4)
        {
            return;
        }
        constexpr int kMaxN = 64;
        if (n > kMaxN)
        {
            return;
        }

        auto cost_of = [&](const std::vector<node_t>& r) {
            return get_route_distance(cvrp, r);
        };

        int passes = 0;
        constexpr int kMaxPasses = 8;
        bool improved = true;
        while (improved && passes < kMaxPasses)
        {
            improved = false;
            ++passes;
            double best_cost = cost_of(cities);
            std::vector<node_t> best = cities;

            for (int i = 1; i < n - 2; ++i)
            {
                for (int j = i + 1; j < n - 1; ++j)
                {
                    for (int k = j + 1; k < n; ++k)
                    {
                        // Segments: [0,i) | [i,j) | [j,k) | [k,n)
                        auto build = [&](bool rev1, bool rev2, bool swap) {
                            std::vector<node_t> out;
                            out.reserve(n);
                            out.insert(
                                out.end(), cities.begin(), cities.begin() + i);
                            std::vector<node_t> s1(
                                cities.begin() + i, cities.begin() + j);
                            std::vector<node_t> s2(
                                cities.begin() + j, cities.begin() + k);
                            if (rev1)
                            {
                                std::reverse(s1.begin(), s1.end());
                            }
                            if (rev2)
                            {
                                std::reverse(s2.begin(), s2.end());
                            }
                            if (!swap)
                            {
                                out.insert(out.end(), s1.begin(), s1.end());
                                out.insert(out.end(), s2.begin(), s2.end());
                            }
                            else
                            {
                                out.insert(out.end(), s2.begin(), s2.end());
                                out.insert(out.end(), s1.begin(), s1.end());
                            }
                            out.insert(
                                out.end(), cities.begin() + k, cities.end());
                            return out;
                        };

                        // Skip pure identity; 2-opt-only reverses are cheap to
                        // re-check; swap variants are the new 3-opt value.
                        for (int mask = 0; mask < 8; ++mask)
                        {
                            const bool rev1 = mask & 1;
                            const bool rev2 = mask & 2;
                            const bool swap = mask & 4;
                            if (!swap && !rev1 && !rev2)
                            {
                                continue;
                            }
                            auto cand = build(rev1, rev2, swap);
                            const double c = cost_of(cand);
                            if (c + 1e-12 < best_cost)
                            {
                                best_cost = c;
                                best = std::move(cand);
                                improved = true;
                            }
                        }
                    }
                }
            }
            if (improved)
            {
                cities = std::move(best);
            }
        }
    }

    void Solver::polish_route(
        const CVRP& cvrp,
        std::vector<node_t>& cities) const
    {
        const int sz = static_cast<int>(cities.size());
        if (sz <= 2)
        {
            return;
        }
        std::vector<node_t> tour(sz);
        tsp_2OPT(cvrp, cities, tour, sz);
        if (this->ls >= 1)
        {
            tsp_or_opt(cvrp, cities);
        }
        if (this->ls >= 2)
        {
            tsp_3opt(cvrp, cities);
            // Re-tighten after 3-opt.
            tsp_or_opt(cvrp, cities);
            tsp_2OPT(cvrp, cities, tour, static_cast<int>(cities.size()));
        }
    }

    void Solver::inter_route_relocate(
        const CVRP& cvrp,
        std::vector<std::vector<node_t>>& routes) const
    {
        if (this->ls < 1 || routes.size() < 2)
        {
            return;
        }
        const node_t depot = cvrp.depot();
        const capacity_t Q = cvrp.capacity();

        auto route_load = [&](const std::vector<node_t>& r) {
            capacity_t load = 0;
            for (node_t u : r)
            {
                load += cvrp[u].demand;
            }
            return load;
        };

        int passes = 0;
        constexpr int kMaxPasses = 8;
        bool improved = true;
        while (improved && passes < kMaxPasses)
        {
            improved = false;
            ++passes;
            for (size_t r1 = 0; r1 < routes.size(); ++r1)
            {
                if (routes[r1].empty())
                {
                    continue;
                }
                for (size_t i = 0; i < routes[r1].size(); ++i)
                {
                    const node_t u = routes[r1][i];
                    const capacity_t dem = cvrp[u].demand;
                    const node_t left =
                        (i == 0) ? depot : routes[r1][i - 1];
                    const node_t right = (i + 1 == routes[r1].size())
                        ? depot
                        : routes[r1][i + 1];
                    const double remove_delta =
                        cvrp.distance_fast(left, right)
                        - cvrp.distance_fast(left, u)
                        - cvrp.distance_fast(u, right);

                    for (size_t r2 = 0; r2 < routes.size(); ++r2)
                    {
                        if (r2 == r1)
                        {
                            continue;
                        }
                        if (route_load(routes[r2]) + dem > Q + 1e-9)
                        {
                            continue;
                        }
                        for (size_t p = 0; p <= routes[r2].size(); ++p)
                        {
                            const node_t ins_left =
                                (p == 0) ? depot : routes[r2][p - 1];
                            const node_t ins_right =
                                (p == routes[r2].size()) ? depot
                                                         : routes[r2][p];
                            const double insert_delta =
                                cvrp.distance_fast(ins_left, u)
                                + cvrp.distance_fast(u, ins_right)
                                - cvrp.distance_fast(ins_left, ins_right);
                            if (remove_delta + insert_delta < -1e-12)
                            {
                                routes[r1].erase(routes[r1].begin() + static_cast<long>(i));
                                routes[r2].insert(
                                    routes[r2].begin() + static_cast<long>(p), u);
                                if (routes[r1].empty())
                                {
                                    routes.erase(routes.begin() + static_cast<long>(r1));
                                }
                                improved = true;
                                break;
                            }
                        }
                        if (improved)
                        {
                            break;
                        }
                    }
                    if (improved)
                    {
                        break;
                    }
                }
                if (improved)
                {
                    break;
                }
            }
        }
    }

    std::vector<std::vector<node_t>> Solver::process_2OPT(
        const CVRP&                             cvrp, 
        const std::vector<std::vector<node_t>>& routes) const
    {
        std::vector<std::vector<node_t>> processed_routes;
        int nroutes = routes.size();
        processed_routes.reserve(nroutes);

        for (int i = 0; i < nroutes; ++i) 
        {
            std::vector<node_t> cities = routes[i];
            polish_route(cvrp, cities);
            processed_routes.push_back(std::move(cities));
        }

        return processed_routes;
    }

    void Solver::process_routes(
        const CVRP& cvrp,
        std::vector<std::vector<node_t>>& routes, 
        distance_t& total_cost) const
    {
        auto processed_routes1  = process_tsp_approx(cvrp, routes);
        processed_routes1       = process_2OPT(cvrp, processed_routes1);
        auto processed_routes2  = process_2OPT(cvrp, routes);

        double min_cost = 0;
        for(size_t i = 0; i < routes.size(); i++)
        {
            auto route_cost     = get_route_distance(cvrp, routes[i]);
            auto route_cost1    = get_route_distance(cvrp, processed_routes1[i]);
            auto route_cost2    = get_route_distance(cvrp, processed_routes2[i]); 

            if(std::min(route_cost1, route_cost2) >= route_cost)
            {
                min_cost        += route_cost;
            }
            else if(std::min(route_cost1, route_cost) >= route_cost2)
            {
                routes[i]       = std::move(processed_routes2[i]);
                min_cost        += route_cost2;
            }
            else
            {
                routes[i]       = std::move(processed_routes1[i]);
                min_cost        += route_cost1;   
            }
        }

        if (this->ls >= 1)
        {
            inter_route_relocate(cvrp, routes);
            // Re-polish routes that may have changed.
            for (auto& r : routes)
            {
                polish_route(cvrp, r);
            }
            min_cost = 0;
            for (const auto& r : routes)
            {
                min_cost += get_route_distance(cvrp, r);
            }
        }

        total_cost = min_cost;
    }

    void Solver::get_routes(
        const CVRP&                                 cvrp, 
        const std::vector <std::vector <node_t>>&   mst_adj, 
        const std::vector <node_t>&                 bucket, 
        std::vector <std::vector <node_t>>&         routes, 
        distance_t&                                 cost,
        distance_t                                  cost_bound,
        unsigned                                    seed) const
    {
        const int num_nodes = static_cast<int>(bucket.size());
        routes.clear();
        cost = 0.0;
        if(num_nodes == 1)
        {
            return;
        }

        const node_t depot          = cvrp.depot();
        const node_t depot_index    = 0;
        std::vector<char> visited(num_nodes, 0);
        std::mt19937 rng(seed);

        std::vector<node_t> curr_route;
        curr_route.reserve(16);

        // One neighbor buffer for all DFS frames (avoids per-node alloc).
        std::vector<node_t> neigh_buf;
        neigh_buf.reserve(static_cast<size_t>(num_nodes) * 2);
        // stack entry: (next index to try, start offset in neigh_buf)
        std::vector<std::pair<int, int>> stack;
        stack.reserve(64);

        auto push_frame = [&](node_t v_idx)
        {
            const auto& neigh = mst_adj[v_idx];
            const int start = static_cast<int>(neigh_buf.size());
            neigh_buf.insert(neigh_buf.end(), neigh.begin(), neigh.end());
            std::shuffle(neigh_buf.begin() + start, neigh_buf.end(), rng);
            stack.push_back({static_cast<int>(neigh_buf.size()) - 1, start});
        };

        visited[depot_index] = 1;
        push_frame(depot_index);

        node_t prev_node = depot;
        capacity_t residue_capacity = cvrp.capacity();

        while(!stack.empty()) 
        {
            int index = stack.back().first;
            const int start = stack.back().second;
            bool pushed_child = false;

            while(index >= start) 
            {
                const node_t v_index = neigh_buf[index];
                --index;

                if(visited[v_index])
                {
                    continue;
                }

                visited[v_index] = 1;
                const node_t v = bucket[v_index];

                if (residue_capacity < cvrp[v].demand)
                {
                    routes.push_back(std::move(curr_route));
                    curr_route.clear();
                    curr_route.reserve(16);
                    cost += cvrp.distance_fast(prev_node, depot);
                    residue_capacity = cvrp.capacity();
                    prev_node = depot;
                    if (cost >= cost_bound)
                    {
                        routes.clear();
                        cost = std::numeric_limits<distance_t>::infinity();
                        return;
                    }
                }

                curr_route.push_back(v);
                cost += cvrp.distance_fast(prev_node, v);
                residue_capacity -= cvrp[v].demand;
                prev_node = v;

                if (cost >= cost_bound)
                {
                    routes.clear();
                    cost = std::numeric_limits<distance_t>::infinity();
                    return;
                }

                stack.back().first = index;
                push_frame(v_index);
                pushed_child = true;
                break;
            }

            if (!pushed_child)
            {
                neigh_buf.resize(static_cast<size_t>(start));
                stack.pop_back();
            }
        }

        if(!curr_route.empty()) 
        { 
            routes.push_back(std::move(curr_route));
            cost += cvrp.distance_fast(prev_node, depot); 
            if (cost >= cost_bound)
            {
                routes.clear();
                cost = std::numeric_limits<distance_t>::infinity();
            }
        }
    }

    Solver::Solver(
        const double _alpha, 
        const int _rho,
        const int _topk,
        const int _ls) 
        : alpha(_alpha), rho(_rho), topk(_topk), ls(_ls) 
    {
    }

    Solution Solver::solve(
        const CVRP& cvrp) const
    {
        double maxMB_before_execution = get_curr_rss_mb();
        auto start = std::chrono::high_resolution_clock::now();

        distance_t final_cost   = 0.0;
        std::vector <std::vector<node_t>> final_routes;

        const int num_buckets = static_cast<int>(std::ceil(360.00 / alpha)); 
        std::vector <std::vector<node_t>> buckets(num_buckets);
        create_buckets(cvrp, buckets);

        int nonempty = 0;
        int max_bucket = 0;
        long long sum_bucket = 0;
        for (int i = 0; i < num_buckets; ++i)
        {
            const int sz = static_cast<int>(buckets[i].size());
            if (sz > 1)
            {
                ++nonempty;
                sum_bucket += sz;
                if (sz > max_bucket)
                {
                    max_bucket = sz;
                }
            }
        }

        std::cerr << "[bpmds] buckets=" << num_buckets
                  << " nonempty=" << nonempty
                  << " max_bucket_size=" << max_bucket
                  << " avg_bucket_size=" << (nonempty ? (double)sum_bucket / nonempty : 0.0)
                  << " threads=" << omp_get_max_threads()
                  << " rho=" << rho
                  << " topk=" << topk
                  << " ls=" << ls
#ifdef BPMDS_USE_CUDA
                  << " cuda=" << (cuda_enabled() ? 1 : 0)
                  << " cuda_prim_min=" << env_int("BPMDS_CUDA_PRIM_MIN", 512)
                  << " cuda_2opt_min=" << env_int("BPMDS_CUDA_2OPT_MIN", 128)
#endif
                  << "\n" << std::flush;

        const uint32_t seed_base = 0x9e3779b9u;
        std::atomic<int> buckets_done{0};
        const int log_every = std::max(1, nonempty / 20);

        // Nested OpenMP: buckets outer, rho trials inner.
        #pragma omp parallel for schedule(dynamic, 1)
        for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id)
        {
            const std::vector<node_t>& bucket = buckets[bucket_id];
            if (bucket.size() <= 1)
            {
                continue;
            }

            // Keep top-K DFS tours, then pick best *after* 2-opt/TSP polish.
            // Pre-LS DFS cost ≠ post-LS cost; ranking only on DFS left gap on the table.
            // K=1 ≡ original single-winner path; larger K improves gap but loosens DFS prune.
            const int kTopK = std::max(1, this->topk);
            struct DfsCand
            {
                distance_t dfs_cost = DBL_MAX;
                std::vector<std::vector<node_t>> routes;
            };
            std::vector<DfsCand> topk_pool;
            topk_pool.reserve(static_cast<size_t>(kTopK));

            std::vector<std::vector<node_t>> mst_adj(bucket.size());
            construct_mst(cvrp, bucket, mst_adj);

            auto consider = [&](std::vector<DfsCand>& pool, distance_t c,
                                std::vector<std::vector<node_t>>& routes)
            {
                if (!(c < DBL_MAX))
                {
                    return;
                }
                if (static_cast<int>(pool.size()) < kTopK)
                {
                    pool.push_back(DfsCand{c, std::move(routes)});
                    return;
                }
                auto worst = std::max_element(
                    pool.begin(), pool.end(),
                    [](const DfsCand& a, const DfsCand& b)
                    {
                        return a.dfs_cost < b.dfs_cost;
                    });
                if (c < worst->dfs_cost)
                {
                    worst->dfs_cost = c;
                    worst->routes = std::move(routes);
                }
            };

            #pragma omp parallel
            {
                std::vector<DfsCand> local_topk;
                local_topk.reserve(static_cast<size_t>(kTopK));
                // Tight DFS prune on *best* so far (as in K=1). Admission to the
                // polish pool still uses worst-of-K; do not loosen get_routes abort.
                distance_t dfs_prune = DBL_MAX;

                #pragma omp for nowait schedule(static)
                for (int trial = 0; trial < rho; ++trial)
                {
                    std::vector<std::vector<node_t>> routes;
                    distance_t cost = 0.0;
                    const unsigned seed =
                        seed_base ^ (static_cast<unsigned>(bucket_id) * 0x85ebca6bu)
                                   ^ (static_cast<unsigned>(trial) * 0xc2b2ae35u);
                    get_routes(cvrp, mst_adj, bucket, routes, cost, dfs_prune, seed);
                    if (!(cost < DBL_MAX))
                    {
                        continue;
                    }
                    if (cost < dfs_prune)
                    {
                        dfs_prune = cost;
                    }
                    distance_t admit = DBL_MAX;
                    if (static_cast<int>(local_topk.size()) >= kTopK)
                    {
                        admit = 0.0;
                        for (const auto& cand : local_topk)
                        {
                            if (cand.dfs_cost > admit)
                            {
                                admit = cand.dfs_cost;
                            }
                        }
                    }
                    if (static_cast<int>(local_topk.size()) < kTopK || cost < admit)
                    {
                        consider(local_topk, cost, routes);
                    }
                }

                #pragma omp critical
                {
                    for (auto& cand : local_topk)
                    {
                        consider(topk_pool, cand.dfs_cost, cand.routes);
                    }
                }
            }

            std::vector<std::vector<node_t>> low_cost_routes;
            distance_t low_cost = DBL_MAX;
            for (auto& cand : topk_pool)
            {
                if (cand.routes.empty())
                {
                    continue;
                }
                distance_t polished = cand.dfs_cost;
                process_routes(cvrp, cand.routes, polished);
                if (polished < low_cost)
                {
                    low_cost = polished;
                    low_cost_routes = std::move(cand.routes);
                }
            }

            if (!low_cost_routes.empty())
            {
                #pragma omp critical
                {
                    for (auto& route : low_cost_routes)
                    {
                        final_routes.push_back(std::move(route));
                    }
                    final_cost += low_cost;
                }
            }

            const int done = ++buckets_done;
            if (done == 1 || done == nonempty || done % log_every == 0)
            {
                const auto now = std::chrono::high_resolution_clock::now();
                const double elapsed = std::chrono::duration<double>(now - start).count();
                #pragma omp critical(bpmds_log)
                {
                    std::cerr << "[bpmds] progress " << done << "/" << nonempty
                              << " buckets (" << std::fixed << std::setprecision(1)
                              << (100.0 * done / std::max(1, nonempty)) << "%)"
                              << " elapsed=" << std::setprecision(2) << elapsed << "s"
                              << " last_bucket=" << bucket_id
                              << " size=" << bucket.size()
                              << "\n" << std::flush;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double maxMB_after_execution = get_peak_rss_mb();
        double execution_time = std::chrono::duration<double>(end - start).count();
        double maxMB_difference = maxMB_after_execution - maxMB_before_execution;

        std::cerr << "[bpmds] solve finished in " << std::fixed << std::setprecision(4)
                  << execution_time << "s"
                  << " cost=" << final_cost
                  << " routes=" << final_routes.size()
                  << " mem_delta_mb=" << maxMB_difference
                  << "\n" << std::flush;

        return Solution(execution_time, maxMB_difference, final_cost, final_routes);
    }
}
