import io
import cProfile
import pstats

class ProfileContext:
    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        return self

    def __exit__(self, *args):
        self.profiler.disable()

        if False:
            ss = io.StringIO()

            ps = pstats.Stats(self.profiler, stream=ss).sort_stats("cumulative")
            ps.print_stats()

            profile = ss.getvalue()
            print(profile)
